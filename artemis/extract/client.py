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
from typing import Any, Literal, Optional, Sequence

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

                self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
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
        self, page: PageDocument, anchor_names: Sequence[str] = ()
    ) -> list[Extraction]:
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
            key = "|".join(
                [
                    prompts.PROMPT_VERSION,
                    self.s.extraction_model,
                    page.text_sha256,
                    str(lead),
                    ",".join(anchors),
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
                page, rosters, self.log, anchor=anchors[0] if anchors else None
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
