"""Claude API wrapper: per-page batching, response cache keyed by (content_hash, prompt_version).

One call per page (or per ~150-sentence chunk of a long page), never one per
sentence. Extraction runs on a cheap fast model; identity adjudication and pivot
verification run on the stronger one and are called a handful of times per job.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

from artemis.config import Settings
from artemis.extract import prompts
from artemis.extract.grounding import degraded_extract, ground, ground_co_listings
from artemis.models import (
    Extraction,
    PageDocument,
    RawCoListing,
    RawExtraction,
    content_hash,
)
from artemis.runtime import BudgetLedger, JobLog
from artemis.scrape.cache import DiskCache

#: Long pages are chunked. Kept small because a 400-sentence Wikipedia article
#: in one call produced enough thinking to exhaust max_tokens before any text
#: was emitted — the model returned stop_reason=max_tokens with empty content.
_CHUNK_SENTENCES = 80
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Models that reject `output_config.effort` outright; learned at runtime so a
#: model released after this build degrades instead of failing every call.
_NO_EFFORT: set[str] = set()

Answer = Literal["yes", "no", "unknown"]


@dataclass
class Verdict:
    same_person: Answer
    reason: str = ""

    @property
    def is_yes(self) -> bool:
        return self.same_person == "yes"


class ClaudeClient:
    """Thin wrapper. Returns data or None; never raises into the crawl loop."""

    def __init__(
        self,
        settings: Settings,
        cache: DiskCache,
        ledger: BudgetLedger,
        log: JobLog,
        *,
        client: Any = None,
    ) -> None:
        self.s = settings
        self.cache = cache
        self.ledger = ledger
        self.log = log
        self._client = client
        if self._client is None and settings.anthropic_api_key:
            try:
                from anthropic import AsyncAnthropic

                # Explicit timeout. The SDK default is 10 minutes and this code
                # retries three times, so one stuck request buys up to half an
                # hour of total silence — which is exactly what it looked like:
                # a live job, 0.1% CPU, no log line for eight minutes, and a
                # budget counter three calls ahead of the completions.
                self._client = AsyncAnthropic(
                    api_key=settings.anthropic_api_key,
                    timeout=settings.claude_timeout_s,
                    max_retries=0,  # this module does its own bounded retrying
                )
            except Exception as exc:  # pragma: no cover - import/config failure
                self.log.warn("claude.unavailable", f"{type(exc).__name__}: {exc}")
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # -- transport ----------------------------------------------------------
    async def _call_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
        cache_key: str,
        effort: str = "",
    ) -> Optional[dict]:
        cached = await self.cache.get("claude", cache_key)
        if cached is not None:
            return cached

        if self._client is None:
            return None
        if not self.ledger.try_spend_claude():
            self.log.warn("claude.skipped", "budget exhausted: max_claude_calls")
            return None

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if effort and model not in _NO_EFFORT:
            output_config["effort"] = effort

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # The system prompt is byte-identical across every page in a job.
            # (Below Haiku 4.5's 4096-token cache minimum today, so this is a
            # no-op until the prompt grows; harmless and correct to declare.)
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }

        resp = None
        for attempt in range(3):
            try:
                resp = await self._client.messages.create(**kwargs)
                break
            except TypeError:
                kwargs.pop("output_config", None)  # SDK too old for structured outputs
                continue
            except Exception as exc:
                name, detail = type(exc).__name__, str(exc)
                # An effort rejection is our request's fault, not the caller's:
                # drop it, remember the model, and retry the same request.
                if "effort" in detail.lower() and "effort" in output_config:
                    _NO_EFFORT.add(model)
                    output_config.pop("effort", None)
                    self.log("claude.effort_unsupported", model)
                    continue
                if "BadRequest" in name and "output_config" in kwargs:
                    kwargs.pop("output_config", None)
                    continue
                retryable = any(k in name for k in ("RateLimit", "APIStatus", "APIConnection",
                                                    "InternalServer", "Overloaded", "Timeout"))
                self.log.warn("claude.error", f"{name}: {detail[:200]}", attempt=attempt,
                              model=model)
                if not retryable or attempt == 2:
                    return None
                await asyncio.sleep(2**attempt)

        if resp is None:
            self.log.warn("claude.no_response", "all attempts exhausted", model=model)
            return None

        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            self.log.warn("claude.refusal", "model declined", model=model)
            return None

        text = "".join(
            getattr(b, "text", "") for b in getattr(resp, "content", []) or []
            if getattr(b, "type", "") == "text"
        ).strip()

        usage = getattr(resp, "usage", None)
        self.log(
            "claude.call",
            f"{model} stop={stop} chars={len(text)}",
            model=model,
            stop_reason=stop,
            output_tokens=getattr(usage, "output_tokens", None),
        )

        if not text:
            # Almost always thinking consuming the whole budget before any text
            # was emitted. Silent None here is what made a whole run look hung.
            self.log.warn(
                "claude.empty",
                f"no text returned (stop_reason={stop}); "
                "raise max_tokens, lower effort, or shrink the chunk",
                model=model,
                max_tokens=max_tokens,
            )
            return None
        if stop == "max_tokens":
            self.log.warn("claude.truncated", "response hit max_tokens; JSON may be cut off",
                          model=model, max_tokens=max_tokens)

        data = _loads(text)
        if data is None:
            self.log.warn("claude.unparseable", text[:200], model=model)
            return None

        await self.cache.set("claude", cache_key, data)
        return data

    # -- extraction ---------------------------------------------------------
    async def extract_page(
        self,
        page: PageDocument,
        anchor_names: Sequence[str] = (),
        *,
        is_known: Optional[Callable[[str], bool]] = None,
    ) -> list[Extraction]:
        """`is_known` reports whether a name is already in the caller's graph.

        Used only by co-listing, to keep a roster that names someone we know
        even when it does not name the anchor.
        """
        if not page.sentences:
            return []
        if not self.enabled:
            return degraded_extract(page, self.log)

        anchors = sorted({a for a in anchor_names if a})
        results: list[Extraction] = []
        seen: set[tuple[int, str, str]] = set()

        for chunk_start in range(0, len(page.sentences), _CHUNK_SENTENCES):
            lead = max(0, chunk_start - self.s.context_window_sentences)
            chunk = page.sentences[lead : chunk_start + _CHUNK_SENTENCES]
            if not chunk:
                continue

            boundary = prompts.make_boundary()
            user = prompts.render_extraction_user_message(
                sentences=[s.text for s in chunk],
                boundary=boundary,
                url=page.final_url,
                title=page.title,
                anchor_names=anchors,
                first_index=lead,
            )
            # Anchors are deliberately NOT in the cache key. They are a recall
            # hint the prompt explicitly says licenses nothing, so the same page
            # yields materially the same extractions whoever we arrived via —
            # but including them meant Wikipedia's Larry Ellison page paid for a
            # fresh extraction once per anchor set that reached it.
            #
            # Safe because anchors are re-applied at *grounding* time, not read
            # from the cache: co-listing still gates on the current anchor, so a
            # cached roster is only admitted for someone actually listed on it.
            key = "|".join(
                [
                    prompts.PROMPT_VERSION,
                    self.s.extraction_model,
                    page.text_sha256,
                    str(lead),
                ]
            )
            data = await self._call_json(
                model=self.s.extraction_model,
                system=prompts.EXTRACTION_SYSTEM_PROMPT,
                user=user,
                schema=prompts.EXTRACTION_JSON_SCHEMA,
                max_tokens=self.s.extraction_max_tokens,
                cache_key=key,
                effort=self.s.extraction_effort,
            )
            if not data:
                continue

            raws: list[RawExtraction] = []
            for item in data.get("extractions") or []:
                try:
                    raws.append(RawExtraction.model_validate(item))
                except Exception as exc:
                    self.log.warn("extraction.malformed", str(exc).splitlines()[0], url=page.url)

            rosters: list[RawCoListing] = []
            for item in data.get("co_listings") or []:
                try:
                    rosters.append(RawCoListing.model_validate(item))
                except Exception as exc:
                    self.log.warn("colisting.malformed", str(exc).splitlines()[0], url=page.url)

            grounded = ground(page, raws, self.log) + ground_co_listings(
                page,
                rosters,
                self.log,
                anchor=anchors[0] if anchors else None,
                is_known=is_known,
            )
            for extraction in grounded:
                fingerprint = (
                    extraction.span_start,
                    extraction.subject_name.casefold(),
                    extraction.object_name.casefold(),
                )
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    results.append(extraction)

        return results

    # -- identity assists ---------------------------------------------------
    async def _verdict(
        self, *, system: str, user: str, cache_key: str, model: str
    ) -> Verdict:
        data = await self._call_json(
            model=model,
            system=system,
            user=user,
            schema=prompts.IDENTITY_JSON_SCHEMA,
            max_tokens=self.s.verification_max_tokens,
            cache_key=cache_key,
        )
        if not data:
            return Verdict("unknown", "no adjudication available")
        answer = str(data.get("same_person", "unknown")).lower()
        if answer not in ("yes", "no", "unknown"):
            answer = "unknown"
        return Verdict(answer, str(data.get("reason", ""))[:400])  # type: ignore[arg-type]

    async def adjudicate_identity(
        self, name: str, left: dict[str, Any], right: dict[str, Any]
    ) -> Verdict:
        """Borderline merge assist. Advisory only — the ladder is code."""
        if not self.enabled:
            return Verdict("unknown", "degraded: no identity adjudication")
        payload = json.dumps({"name": name, "observation_a": left, "observation_b": right},
                             ensure_ascii=False, sort_keys=True, indent=2)
        return await self._verdict(
            system=prompts.IDENTITY_SYSTEM_PROMPT,
            user=payload,
            cache_key=f"identity|{prompts.PROMPT_VERSION}|{self.s.identity_model}|"
                      f"{content_hash(payload)}",
            model=self.s.identity_model,
        )

    async def classify_fame(self, name: str, context: str = "") -> tuple[str, str]:
        """"famous" or "not_famous", plus one sentence of reasoning.

        Used only to choose which direction the frontier travels. Defaults to
        "not_famous" when unavailable: most people are, and wrongly assuming
        fame sends the crawl into celebrity coverage the person never appears in.
        """
        if not self.enabled:
            return "not_famous", "no classifier available"
        payload = json.dumps({"name": name, "context": context}, ensure_ascii=False,
                             sort_keys=True)
        data = await self._call_json(
            model=self.s.strategy_model,
            system=prompts.FAME_SYSTEM_PROMPT,
            user=payload,
            schema=prompts.FAME_JSON_SCHEMA,
            max_tokens=1000,
            cache_key=f"fame|{prompts.PROMPT_VERSION}|{self.s.strategy_model}|"
                      f"{content_hash(payload)}",
        )
        if not data:
            return "not_famous", "classifier unavailable"
        fame = str(data.get("fame", "not_famous"))
        if fame not in ("famous", "not_famous"):
            fame = "not_famous"
        return fame, str(data.get("why", ""))[:200]

    async def choose_angle(
        self, subject: str, target: str, attributes: dict[str, list[str]]
    ) -> tuple[str, str]:
        """Pick which pre-written query angle to spend the next search on.

        Returns (angle, why). Falls back to "generic" whenever the call is
        unavailable or the answer is unrecognised — a strategy failure must
        degrade to the broad search, never to no search.
        """
        if not self.enabled or not self.s.strategy_enabled:
            return "generic", "strategy selection disabled"
        if not any(attributes.values()):
            # Nothing grounded to reason from; choosing an angle here would be
            # reasoning on top of a guess.
            return "generic", "no grounded attributes for this person"

        payload = json.dumps(
            {"subject": subject, "subject_attributes": attributes, "target": target},
            ensure_ascii=False, sort_keys=True, indent=2,
        )
        data = await self._call_json(
            model=self.s.strategy_model,
            system=prompts.STRATEGY_SYSTEM_PROMPT,
            user=payload,
            schema=prompts.STRATEGY_JSON_SCHEMA,
            max_tokens=1000,
            cache_key=f"strategy|{prompts.PROMPT_VERSION}|{self.s.strategy_model}|"
                      f"{content_hash(payload)}",
        )
        if not data:
            return "generic", "strategy call unavailable"
        angle = str(data.get("angle", "generic"))
        valid = set(prompts.STRATEGY_JSON_SCHEMA["properties"]["angle"]["enum"])
        if angle not in valid:
            angle = "generic"
        return angle, str(data.get("why", ""))[:300]

    async def propose_bridges(
        self,
        subject: str,
        subject_attributes: dict[str, list[str]],
        target: str,
        target_attributes: dict[str, list[str]],
        limit: int,
    ) -> list[tuple[str, str, str]]:
        """[(bridge_name, why, query)] — who might connect these two, and how to check.

        The one place a model writes text that leaves this system. It is bounded
        rather than trusted: the caller sanitises every query, and nothing here
        can become a relationship — a page still has to state it.
        """
        if not self.enabled or not self.s.strategy_enabled or limit <= 0:
            return []
        if not any(subject_attributes.values()) and not any(target_attributes.values()):
            # Nothing grounded on either end. Anything proposed here would be
            # recall of who these names are, not reasoning about the graph.
            return []

        payload = json.dumps(
            {
                "subject": subject, "subject_attributes": subject_attributes,
                "target": target, "target_attributes": target_attributes,
                "max_bridges": limit,
            },
            ensure_ascii=False, sort_keys=True, indent=2,
        )
        data = await self._call_json(
            model=self.s.strategy_model,
            system=prompts.BRIDGES_SYSTEM_PROMPT,
            user=payload,
            schema=prompts.BRIDGES_JSON_SCHEMA,
            max_tokens=2000,
            cache_key=f"bridges|{prompts.PROMPT_VERSION}|{self.s.strategy_model}|"
                      f"{content_hash(payload)}",
        )
        if not data:
            return []

        out: list[tuple[str, str, str]] = []
        blocked = {subject.casefold(), target.casefold()}
        for item in (data.get("bridges") or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            query = str(item.get("query", "")).strip()
            # The bridge is the person between them; proposing either endpoint
            # is a restatement of the question, and its query duplicates
            # DIRECT_BRIDGE, which already runs on the seeds.
            if not name or not query or name.casefold() in blocked:
                continue
            out.append((name, str(item.get("why", ""))[:200], query))
        return out

    async def choose_frontier(
        self,
        candidates: Sequence[dict[str, Any]],
        target: str,
        target_attributes: dict[str, list[str]],
    ) -> tuple[list[str], str]:
        """Rank candidate ids by likelihood of reaching `target`.

        Returns (ordered_ids, why). The caller's order is authoritative on
        anything this does not rank: unknown ids are dropped and omitted ones
        are appended in their original order, so a partial or hallucinated
        answer degrades to the heuristic rather than losing candidates.

        Advisory only. A wrong ranking spends the level on worse candidates; it
        cannot introduce a claim about anyone, because expansion still grounds
        every edge the same way.
        """
        ids = [str(c.get("id", "")) for c in candidates if c.get("id")]
        if len(ids) < 2:
            return ids, "nothing to rank"
        if not self.enabled or not self.s.strategy_enabled:
            return ids, "frontier ranking disabled"
        if not any(target_attributes.values()):
            # Same refusal as choose_angle: with nothing grounded about the
            # target there is nothing to rank proximity against, and the model
            # would fall back to whoever it recognises — which is prominence,
            # the bias this exists to correct.
            return ids, "no grounded attributes for the target"

        payload = json.dumps(
            {"target": target, "target_attributes": target_attributes,
             "candidates": list(candidates)},
            ensure_ascii=False, sort_keys=True, indent=2,
        )
        data = await self._call_json(
            model=self.s.strategy_model,
            system=prompts.FRONTIER_SYSTEM_PROMPT,
            user=payload,
            schema=prompts.FRONTIER_JSON_SCHEMA,
            max_tokens=2000,
            cache_key=f"frontier|{prompts.PROMPT_VERSION}|{self.s.strategy_model}|"
                      f"{content_hash(payload)}",
        )
        if not data:
            return ids, "frontier ranking unavailable"

        allowed = set(ids)
        seen: set[str] = set()
        ordered = [
            str(x) for x in (data.get("ranked") or [])
            if str(x) in allowed and not (str(x) in seen or seen.add(str(x)))
        ]
        ordered += [i for i in ids if i not in seen]
        return ordered, str(data.get("why", ""))[:300]

    async def assess_reachability(
        self,
        candidates: Sequence[dict[str, Any]],
        target: str,
        target_attributes: dict[str, list[str]],
    ) -> list[tuple[str, Answer, str]]:
        """[(id, "yes"|"no"|"unknown", why)] — which candidates lead to `target`.

        Every candidate comes back, in the order given, whatever happens: no
        key, strategy disabled, a failed or malformed call, a batch that hit the
        call cap, an answer naming ids that were never sent. All of those land
        as "unknown", which the caller still pursues — losing a candidate here
        would silently delete a person from the search.

        Batched because the staged strategy asks about ~144 people at once and
        one call each would be 144 calls; `reachability_max_calls` bounds the
        rest. Advisory only, exactly like choose_frontier: this reorders what
        gets searched next and can introduce no claim about anyone.
        """
        ids = [str(c.get("id", "")) for c in candidates if c.get("id")]
        verdicts: dict[str, tuple[Answer, str]] = {}

        def fill(reason: str) -> list[tuple[str, Answer, str]]:
            return [(i, verdicts.get(i, ("unknown", reason))[0],
                     verdicts.get(i, ("unknown", reason))[1]) for i in ids]

        if not ids:
            return []
        if not self.enabled or not self.s.strategy_enabled:
            return fill("reachability assessment disabled")
        if not any(target_attributes.values()):
            # Same refusal as choose_frontier: with nothing grounded about the
            # target there is nothing to judge proximity against, and the model
            # would answer from whoever it recognises — which is prominence,
            # the bias this exists to correct.
            return fill("no grounded attributes for the target")

        by_id = {str(c.get("id", "")): c for c in candidates if c.get("id")}
        size = max(1, self.s.reachability_batch_size)
        batches = [ids[i : i + size] for i in range(0, len(ids), size)]
        allowed = max(0, self.s.reachability_max_calls)
        if len(batches) > allowed:
            self.log.warn(
                "reachability.capped",
                f"{len(ids)} candidates need {len(batches)} calls; "
                f"reachability_max_calls is {allowed} — the rest stay unknown "
                "and are still pursued by rank",
                candidates=len(ids), batches=len(batches), allowed=allowed,
            )
            batches = batches[:allowed]

        for batch in batches:
            payload = json.dumps(
                {"target": target, "target_attributes": target_attributes,
                 "candidates": [by_id[i] for i in batch]},
                ensure_ascii=False, sort_keys=True, indent=2,
            )
            data = await self._call_json(
                model=self.s.strategy_model,
                system=prompts.REACHABILITY_SYSTEM_PROMPT,
                user=payload,
                schema=prompts.REACHABILITY_JSON_SCHEMA,
                max_tokens=4000,
                cache_key=f"reachability|{prompts.PROMPT_VERSION}|{self.s.strategy_model}|"
                          f"{content_hash(payload)}",
            )
            if not data:
                continue  # this batch stays unknown; the others still get answers

            batch_ids = set(batch)
            for item in data.get("assessments") or []:
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("id", ""))
                # An id from another batch, or one the model invented, is not a
                # verdict about anybody we asked about.
                if node_id not in batch_ids or node_id in verdicts:
                    continue
                answer = str(item.get("reaches", "unknown")).lower()
                if answer not in ("yes", "no", "unknown"):
                    answer = "unknown"
                verdicts[node_id] = (answer, str(item.get("why", ""))[:200])  # type: ignore[arg-type]

        return fill("not assessed")

    async def verify_pivot(
        self, name: str, arriving: dict[str, Any], leaving: dict[str, Any]
    ) -> Verdict:
        if not self.enabled:
            return Verdict("unknown", "degraded: no pivot adjudication")
        payload = json.dumps({"pivot_name": name, "arriving_source": arriving,
                              "leaving_source": leaving},
                             ensure_ascii=False, sort_keys=True, indent=2)
        return await self._verdict(
            system=prompts.PIVOT_SYSTEM_PROMPT,
            user=payload,
            cache_key=f"pivot|{prompts.PROMPT_VERSION}|{self.s.verification_model}|"
                      f"{content_hash(payload)}",
            model=self.s.verification_model,
        )


def _loads(text: str) -> Optional[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if not m:
            return None
        try:
            value = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
