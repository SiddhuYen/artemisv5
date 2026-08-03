"""Ground ArtemisV2's seed-connection CSVs into the V5 relation store.

The CSVs (artemisv2 PR #48 and its ancestors) are 1,270 curated
person-to-person connections across tech, VC, finance, celebrity and sport.
They are *claims*, not evidence: `evidence_note` is a hand-written summary, not
a verbatim span, there are no character offsets, and `relationship_type` is edge
typing that this system deliberately does not have. Writing them straight into
the store would put unverifiable assertions behind hops the output presents as
page-grounded.

So they are used as a worklist instead. Each row contributes two things V5
cannot cheaply get on its own: a known-good source URL, and a name pair worth
looking for. Every page is fetched and extracted through the ordinary pipeline,
so whatever lands in the store carries a byte-exact span at real offsets on a
real URL — and pages that do not actually assert the curated relationship
contribute nothing, which is the point.

The unsupported pairs are reported rather than discarded: they say where the
seed data asserts something its own cited source does not.

    python scripts/import_seed_connections.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artemis.config import Settings  # noqa: E402
from artemis.extract.client import ClaudeClient  # noqa: E402
from artemis.graph.relations import RelationCache  # noqa: E402
from artemis.identity.normalize import could_be_same_name  # noqa: E402
from artemis.runtime import BudgetLedger, JobLog  # noqa: E402
from artemis.scrape.cache import DiskCache  # noqa: E402
from artemis.scrape.fetcher import Fetcher  # noqa: E402

V2_REPO = Path("/Users/siddhu/Pantheon-Artemis/artemisv2")
V2_REF = "refs/remotes/origin/pr48"


@dataclass
class Stats:
    urls_total: int = 0
    urls_fetched: int = 0
    urls_failed: int = 0
    pages_extracted: int = 0
    relations_written: int = 0
    pairs_total: int = 0
    pairs_confirmed: int = 0
    unsupported: list[tuple[str, str, str]] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)


def load_seed_rows() -> list[dict[str, str]]:
    """Read every seed CSV out of the v2 PR ref (no checkout, no working-tree change)."""
    listing = subprocess.run(
        ["git", "ls-tree", "-r", V2_REF, "--name-only"],
        cwd=V2_REPO, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    files = [f for f in listing if "seed" in f.lower() and f.endswith(".csv")]

    rows: list[dict[str, str]] = []
    for path in files:
        blob = subprocess.run(
            ["git", "show", f"{V2_REF}:{path}"],
            cwd=V2_REPO, capture_output=True, text=True, check=True,
        ).stdout
        for row in csv.DictReader(blob.splitlines()):
            url = (row.get("source_url") or "").strip()
            a = (row.get("target_name") or "").strip()
            b = (row.get("connection_name") or "").strip()
            if url.startswith("http") and a and b:
                rows.append({"a": a, "b": b, "url": url,
                             "kind": (row.get("relationship_type") or "").strip(),
                             "file": Path(path).name})
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap URLs (0 = all)")
    # Deliberately modest. This runs for tens of minutes alongside whatever
    # you are doing with the service, and sharing an API key, a network and a
    # SQLite file with a live job matters more here than finishing sooner.
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_seed_rows()
    by_url: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_url[row["url"]].append(row)
    urls = list(by_url)
    if args.limit:
        urls = urls[: args.limit]

    st = Stats(urls_total=len(urls), pairs_total=sum(len(by_url[u]) for u in urls))
    print(f"seed rows: {len(rows)} | unique URLs: {len(by_url)} | importing: {len(urls)}",
          flush=True)
    if args.dry_run:
        for u in urls[:10]:
            print(f"  {u}  ({len(by_url[u])} pairs)", flush=True)
        return 0

    # Wikipedia carries ~all of these; a 1s/domain delay would serialise the run
    # into hours. 0.25s across 4 connections is still well inside their limits.
    settings = Settings(
        per_domain_concurrency=3,
        per_domain_delay_s=0.3,
        fetch_concurrency=4,
        extraction_concurrency=3,
    )
    log = JobLog()
    ledger = BudgetLedger(
        max_serper_credits=0,          # no search: every URL is already known
        max_fetches=len(urls) + 100,
        max_claude_calls=20_000,
        max_nodes_expanded=0,
        wall_clock_s=6 * 3600,
    )
    cache = DiskCache(settings.cache_dir, enabled=True, ttl_s=settings.cache_ttl_s)
    relations = RelationCache(settings.cache_dir, ttl_s=settings.relation_cache_ttl_s)
    claude = ClaudeClient(settings, cache, ledger, log)
    if not claude.enabled:
        print("ERROR: no Anthropic key — extraction would be degraded", flush=True)
        return 1

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async with Fetcher(settings, cache, ledger, log) as fetcher:

        async def handle(url: str) -> None:
            nonlocal done
            pairs = by_url[url]
            async with sem:
                outcome = await fetcher.fetch(url)
                if outcome.page is None:
                    st.urls_failed += 1
                else:
                    st.urls_fetched += 1
                    page = outcome.page
                    anchors = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
                    extractions = await claude.extract_page(page, anchors)
                    st.pages_extracted += 1
                    for ex in extractions:
                        await relations.add(
                            ex,
                            source_url=page.final_url,
                            source_title=page.title,
                            retrieved_at=page.retrieved_at,
                        )
                    st.relations_written += len(extractions)

                    # Did the page actually assert each curated pair?
                    for pair in pairs:
                        if any(
                            (could_be_same_name(pair["a"], e.subject_name)
                             and could_be_same_name(pair["b"], e.object_name))
                            or (could_be_same_name(pair["a"], e.object_name)
                                and could_be_same_name(pair["b"], e.subject_name))
                            for e in extractions
                        ):
                            st.pairs_confirmed += 1
                        else:
                            st.unsupported.append((pair["a"], pair["b"], url))

            done += 1
            if done % 25 == 0 or done == len(urls):
                mins = (time.monotonic() - st.started) / 60
                print(
                    f"[{done:4}/{len(urls)}] {mins:5.1f}m | fetched {st.urls_fetched} "
                    f"failed {st.urls_failed} | relations {st.relations_written} "
                    f"| pairs confirmed {st.pairs_confirmed}/{st.pairs_total} "
                    f"| claude {ledger.claude_calls}",
                    flush=True,
                )

        await asyncio.gather(*(handle(u) for u in urls), return_exceptions=True)

    summary = await relations.stats()
    report = {
        "urls_total": st.urls_total,
        "urls_fetched": st.urls_fetched,
        "urls_failed": st.urls_failed,
        "relations_written": st.relations_written,
        "pairs_total": st.pairs_total,
        "pairs_confirmed": st.pairs_confirmed,
        "pairs_unsupported": len(st.unsupported),
        "claude_calls": ledger.claude_calls,
        "elapsed_min": round((time.monotonic() - st.started) / 60, 1),
        "relation_store": summary,
        "unsupported_sample": st.unsupported[:200],
    }
    out = Path("seed_import_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== done ===", flush=True)
    for k, v in report.items():
        if k != "unsupported_sample":
            print(f"  {k}: {v}", flush=True)
    print(f"  report: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
