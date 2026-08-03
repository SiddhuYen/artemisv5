"""Disk cache shared by search, fetch, and Claude responses.

Keyed by an explicit namespace plus a caller-supplied key, hashed into a
sharded path. Dev re-runs against the same query set cost nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    def __init__(self, root: Path, *, enabled: bool = True, ttl_s: int = 604_800) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.ttl_s = ttl_s
        self.hits = 0
        self.misses = 0

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / digest[:2] / digest[2:4] / f"{digest}.json"

    def _read(self, path: Path) -> Optional[Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if self.ttl_s and time.time() - raw.get("stored_at", 0) > self.ttl_s:
            return None
        return raw.get("value")

    def _write(self, path: Path, value: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"stored_at": time.time(), "value": value}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            pass  # cache failures are never fatal

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        value = await asyncio.to_thread(self._read, self._path(namespace, key))
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    async def set(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        await asyncio.to_thread(self._write, self._path(namespace, key), value)
