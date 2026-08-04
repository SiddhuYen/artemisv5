"""Async HTTP fetch: bounded concurrency, per-domain rate limit, robots.txt, retry.

Individual dead URLs, parse errors, and rate limits log a warning and the crawl
continues — a fetch layer that raises would let one bad host kill a job.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from artemis.config import Settings
from artemis.models import PageDocument, utcnow
from artemis.runtime import BudgetLedger, JobLog
from artemis.scrape.cache import DiskCache
from artemis.scrape.extract_text import build_document

_BINARY_SUFFIXES = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".gz",
    ".tar", ".rar", ".7z", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".ico", ".mp3", ".mp4", ".mov", ".avi", ".wav", ".exe", ".dmg", ".pkg",
    ".woff", ".woff2", ".ttf", ".css", ".js", ".json", ".xml", ".rss",
)
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

#: Hosts that never yield a usable page, so every attempt costs a robots.txt
#: request, a per-domain delay, and a fetch slot for nothing. Measured, not
#: guessed: across this session's runs linkedin.com wasted 19 attempts,
#: facebook.com 19, instagram.com 18, podcasters.spotify.com 17.
#:
#: Deliberately limited to structurally hostile hosts — social networks, contact
#: scrapers, and podcast hosts that refuse crawlers. Corporate sites with strict
#: robots (oracle.com) are NOT here: they block some paths and serve others, and
#: a blocklist entry is permanent while a robots rule is per-path.
DEFAULT_BLOCKED_DOMAINS = frozenset({
    # Social — robots-disallowed to crawlers, and login-walled besides
    "linkedin.com", "facebook.com", "instagram.com", "threads.net",
    "x.com", "twitter.com", "tiktok.com", "pinterest.com", "reddit.com",
    "quora.com", "youtube.com",
    # Contact/lead scrapers — no relationship prose, often paywalled
    "rocketreach.co", "zoominfo.com", "signalhire.com", "lusha.com",
    "apollo.io", "contactout.com", "leadiq.com", "hunter.io",
    # Aggregators that block or render nothing useful
    "crunchbase.com", "tracxn.com", "glassdoor.com", "indeed.com",
    "academia.edu", "researchgate.net", "scribd.com",
    # Podcast hosts observed refusing crawlers (others — omny.fm, acast,
    # spreaker, podcasts.apple.com — do serve, so they stay off this list)
    "podcasters.spotify.com", "redcircle.com", "anchor.fm",
})


@dataclass
class FetchOutcome:
    url: str
    page: Optional[PageDocument] = None
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.page is not None


class Fetcher:
    def __init__(
        self,
        settings: Settings,
        cache: DiskCache,
        ledger: BudgetLedger,
        log: JobLog,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.s = settings
        self.cache = cache
        self.ledger = ledger
        self.log = log
        self._client = client
        self._owns_client = client is None
        self._global_sem = asyncio.Semaphore(settings.fetch_concurrency)
        self._domain_sems: dict[str, asyncio.Semaphore] = {}
        self._domain_last: dict[str, float] = {}
        self._robots: dict[str, Optional[RobotFileParser]] = {}
        self._robots_lock = asyncio.Lock()
        # Parsing gets its own pool, sized by parse_concurrency (default 1).
        #
        # trafilatura, readability-lxml and BeautifulSoup(html, "lxml") all end
        # up in libxml2, and asyncio.to_thread put up to fetch_concurrency of
        # them in flight at once on the shared default executor. That is what
        # killed the container: `double free or corruption (out)`, SIGSEGV,
        # exit 139 — a native heap corruption, not a Python exception, so
        # nothing in the job log records it and the run simply vanishes.
        #
        # One worker still keeps the CPU-bound work off the event loop, which is
        # the whole reason it was moved to a thread; it just stops two of them
        # being inside libxml2 together. On a one-vCPU instance that costs
        # essentially nothing, because the parses were contending for one core
        # anyway.
        self._parse_pool = ThreadPoolExecutor(
            max_workers=max(1, int(getattr(settings, "parse_concurrency", 1) or 1)),
            thread_name_prefix="artemis-parse",
        )
        self.blocked_domains = frozenset(
            (DEFAULT_BLOCKED_DOMAINS if settings.use_default_blocklist else frozenset())
            | {d.strip().lower() for d in settings.blocked_domains_extra.split(",") if d.strip()}
        )
        self.blocked_skips = 0

    def _is_blocked(self, url: str) -> bool:
        """Host, or any parent domain, on the blocklist (so in.linkedin.com too)."""
        host = urlparse(url).netloc.lower().split(":")[0].removeprefix("www.")
        parts = host.split(".")
        return any(
            ".".join(parts[i:]) in self.blocked_domains for i in range(len(parts) - 1)
        )

    # -- lifecycle ----------------------------------------------------------
    async def __aenter__(self) -> "Fetcher":
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.s.fetch_timeout_s,
                headers={
                    "User-Agent": self.s.user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                limits=httpx.Limits(max_connections=self.s.fetch_concurrency * 2),
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
        # Do not wait on in-flight parses: a cancelled crawl is already
        # unwinding, and a stuck libxml2 call would hold the whole shutdown.
        self._parse_pool.shutdown(wait=False, cancel_futures=True)

    # -- politeness ---------------------------------------------------------
    def _domain_sem(self, host: str) -> asyncio.Semaphore:
        if host not in self._domain_sems:
            self._domain_sems[host] = asyncio.Semaphore(self.s.per_domain_concurrency)
        return self._domain_sems[host]

    async def _respect_delay(self, host: str) -> None:
        last = self._domain_last.get(host)
        if last is not None:
            wait = self.s.per_domain_delay_s - (time.monotonic() - last)
            if wait > 0:
                await asyncio.sleep(wait)
        self._domain_last[host] = time.monotonic()

    async def _robots_ok(self, url: str) -> bool:
        if not self.s.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        async with self._robots_lock:
            if origin not in self._robots:
                parser: Optional[RobotFileParser] = None
                try:
                    assert self._client is not None
                    resp = await self._client.get(f"{origin}/robots.txt", timeout=8.0)
                    if resp.status_code < 400 and resp.text:
                        parser = RobotFileParser()
                        parser.parse(resp.text.splitlines())
                except Exception:
                    parser = None  # unreachable robots.txt is not a prohibition
                self._robots[origin] = parser
            parser = self._robots[origin]
        if parser is None:
            return True
        try:
            return parser.can_fetch(self.s.user_agent, url)
        except Exception:
            return True

    # -- fetching -----------------------------------------------------------
    async def fetch_many(self, urls: list[str]) -> list[FetchOutcome]:
        seen: set[str] = set()
        ordered: list[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        return list(await asyncio.gather(*(self.fetch(u) for u in ordered)))

    async def fetch(self, url: str) -> FetchOutcome:
        if any(urlparse(url).path.lower().endswith(sfx) for sfx in _BINARY_SUFFIXES):
            return FetchOutcome(url, reason="binary")

        # Checked before everything else: no budget, no robots.txt round trip,
        # no per-domain delay spent on a host that has never served us a page.
        if self._is_blocked(url):
            self.blocked_skips += 1
            return FetchOutcome(url, reason="blocked_domain")

        cached = await self.cache.get("fetch", url)
        if cached is not None:
            return await self._to_outcome(url, cached, from_cache=True)

        if not self.ledger.try_spend_fetch():
            return FetchOutcome(url, reason="budget")

        if not await self._robots_ok(url):
            self.log("fetch.skipped", "robots.txt disallows", url=url)
            return FetchOutcome(url, reason="robots")

        host = urlparse(url).netloc
        async with self._global_sem, self._domain_sem(host):
            await self._respect_delay(host)
            payload = await self._get_with_retry(url)

        if payload is None:
            return FetchOutcome(url, reason="unreachable")

        await self.cache.set("fetch", url, payload)
        return await self._to_outcome(url, payload, from_cache=False)

    async def _get_with_retry(self, url: str) -> Optional[dict]:
        assert self._client is not None
        delay = 1.0
        for attempt in range(self.s.fetch_max_retries + 1):
            try:
                resp = await self._client.get(url)
            except Exception as exc:
                self.log.warn("fetch.error", f"{type(exc).__name__}: {exc}", url=url,
                              attempt=attempt)
                if attempt >= self.s.fetch_max_retries:
                    return None
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if resp.status_code in _RETRY_STATUS and attempt < self.s.fetch_max_retries:
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if (retry_after or "").isdigit() else delay
                self.log.warn("fetch.retry", f"HTTP {resp.status_code}", url=url, wait_s=wait)
                await asyncio.sleep(min(wait, 15.0))
                delay *= 2
                continue

            if resp.status_code >= 400:
                self.log.warn("fetch.failed", f"HTTP {resp.status_code}", url=url)
                return None

            ctype = resp.headers.get("content-type", "").lower()
            if ctype and "html" not in ctype and "text/plain" not in ctype:
                self.log("fetch.skipped", f"content-type {ctype}", url=url)
                return None

            html = resp.text[: self.s.fetch_max_bytes]
            return {
                "final_url": str(resp.url),
                "status": resp.status_code,
                "html": html,
                "retrieved_at": utcnow().isoformat(),
            }
        return None

    async def _to_outcome(self, url: str, payload: dict, *, from_cache: bool) -> FetchOutcome:
        # trafilatura/readability parsing is CPU-bound and was running inline on
        # the event loop: a burst of pages stalled every other coroutine,
        # including the API's own responses to /jobs polling.
        doc, reason = await asyncio.get_running_loop().run_in_executor(
            self._parse_pool,
            partial(
                build_document,
                url=url,
                final_url=payload.get("final_url", url),
                html=payload.get("html", ""),
                max_sentences=self.s.max_sentences_per_page,
            ),
        )
        if doc is None:
            self.log("fetch.skipped", f"unusable page: {reason}", url=url)
            return FetchOutcome(url, reason=reason)
        doc.from_cache = from_cache
        self.log(
            "page.fetched",
            doc.title or url,
            url=url,
            final_url=doc.final_url,
            sentences=len(doc.sentences),
            chars=len(doc.text),
            cached=from_cache,
        )
        return FetchOutcome(url, page=doc)
