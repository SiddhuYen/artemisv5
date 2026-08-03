"""Configuration: budgets, concurrency, timeouts, model names — all env-injected.

Every budget is enforced and reported. Nothing here is read at import time by
the domain layer; the app builds one Settings instance and passes it down.

Env names are ARTEMIS_-prefixed, with aliases for the bare names people already
have in their shells (ANTHROPIC_API_KEY, SERPER_API_KEY).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARTEMIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- credentials -------------------------------------------------------
    # Serper is the only door to the web; without it a job fails.
    serper_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ARTEMIS_SERPER_API_KEY", "SERPER_API_KEY"),
    )
    # Without an Anthropic key the extractor degrades to strict same-sentence
    # matching and every result carries WARN_NO_REFERENT_RESOLUTION.
    anthropic_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ARTEMIS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )

    # --- models ------------------------------------------------------------
    # Extraction runs once per page across hundreds of pages: cheap and fast.
    # NOT aliased to ARTEMIS_CLAUDE_EXTRACT_MODEL: in ArtemisV2 that name means
    # the premium extractor (defaults to opus-5). Here it means the per-page
    # workhorse that runs hundreds of times a job. Same name, opposite intent —
    # honouring it silently pointed this extractor at claude-sonnet-5.
    extraction_model: str = "claude-haiku-4-5"
    #: Mechanical copy-out-spans work; low effort keeps thinking from eating
    #: the whole token budget on long pages. Dropped automatically if a model
    #: rejects it.
    extraction_effort: str = "low"
    # Pivot verification runs on a handful of interior nodes at return time.
    verification_model: str = "claude-opus-5"
    # Borderline merge adjudication (ladder rungs 3 and 4) — same tier as pivot.
    identity_model: str = "claude-opus-5"
    extraction_max_tokens: int = 16000
    verification_max_tokens: int = 4000
    strategy_enabled: bool = True
    strategy_model: str = "claude-haiku-4-5"

    # --- search ------------------------------------------------------------
    serper_endpoint: str = "https://google.serper.dev/search"
    serper_results_per_query: int = 10
    serper_gl: str = "us"
    serper_hl: str = "en"
    serper_batch_size: int = 10  # queries per batched POST
    serper_max_retries: int = 3

    # --- fetch -------------------------------------------------------------
    user_agent: str = "ArtemisBot/0.1 (+https://example.invalid/artemis)"
    fetch_concurrency: int = 8
    per_domain_concurrency: int = 2
    per_domain_delay_s: float = 1.0
    fetch_timeout_s: float = 20.0
    fetch_max_retries: int = 2
    fetch_max_bytes: int = 3_000_000
    respect_robots: bool = True

    # --- extraction --------------------------------------------------------
    context_window_sentences: int = 3
    max_sentences_per_page: int = 400
    extraction_concurrency: int = 4

    # --- expansion ---------------------------------------------------------
    # Asymmetric depth budget only: A expands two levels, B one, so routes top
    # out at three hops. The expansion policy itself is symmetric.
    max_depth_a: int = 2
    max_depth_b: int = 1
    frontier_cap_per_level: int = 12
    fetch_top_n_per_node: int = 6
    max_routes_returned: int = 5

    # --- structured providers ----------------------------------------------
    # URL-discovery only: they find documents the open web search misses, which
    # are then fetched and grounded like any other page. Serper remains the only
    # general web-search provider.
    edgar_enabled: bool = True
    #: The SEC requires a User-Agent carrying real contact details.
    edgar_user_agent: str = "Artemis Research research@example.invalid"
    propublica_enabled: bool = True
    opencorporates_api_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ARTEMIS_OPENCORPORATES_API_TOKEN", "OPENCORPORATES_API_TOKEN"
        ),
    )
    #: Discovered URLs fetched per node expansion.
    provider_urls_per_node: int = 6
    #: Nodes per level that get structured lookups. The payoff is concentrated
    #: near the endpoints; running them on the whole frontier fetched dozens of
    #: tangential filings.
    provider_nodes_per_level: int = 4

    #: Drop grounded edges where neither party is already in the graph. They are
    #: real relationships, but they form islands no path can cross.
    require_graph_connection: bool = True

    # --- relation cache ----------------------------------------------------
    #: Every grounded relation, asserted or co-listed, persists across jobs with
    #: its evidence. Later jobs replay them free of search, fetch, and model cost.
    relation_cache_enabled: bool = True
    relation_cache_ttl_s: int = 60 * 60 * 24 * 30

    # --- identity ----------------------------------------------------------
    # Above this many mutually inconsistent attribute clusters for one name,
    # attribute-only merges are refused (the rarity gate).
    common_name_cluster_threshold: int = 3
    drop_name_only_pivots: bool = False

    # --- budgets (defaults; per-request Budget overrides these) -------------
    max_nodes_expanded: int = 60
    # Deliberately NOT aliased to ARTEMIS_SERPER_QUOTA: an account-level quota
    # (e.g. 50,000) is not a per-job ceiling, and reading it as one would let a
    # single runaway job spend the month's credits.
    max_serper_credits: int = 150
    max_fetches: int = 250
    max_claude_calls: int = 200
    wall_clock_s: float = 600.0

    # --- cache -------------------------------------------------------------
    cache_dir: Path = Field(default=Path(".artemis-cache"))
    cache_enabled: bool = True
    cache_ttl_s: int = 60 * 60 * 24 * 7

    # --- server ------------------------------------------------------------
    job_retention_s: int = 60 * 60 * 6
    max_log_entries_per_job: int = 5000

    @property
    def claude_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
