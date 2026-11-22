"""Response cache: one entry per item x question x repeat, so re-runs never re-bill.

Lives under `paths.cache_dir(model, version)`, i.e. `cache_mock/` in mock mode,
so mock answers can never be served as live ones. diskcache ships no type
stubs; the untyped calls stay in this file.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import diskcache  # pyright: ignore[reportMissingTypeStubs]

from recalibench.core.types import Question


def canonical_state(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(
    state: dict[str, Any], question: Question, repeat_idx: int, namespace: str = ""
) -> str:
    """sha256 over everything that changes the answer (version is in the directory)."""
    payload = {
        "state": canonical_state(state),
        "kind": question.kind,
        "text": question.text,
        "options": [[o.key, o.label, o.description] for o in question.options],
        "repeat": repeat_idx,
        "ns": namespace,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._cache: Any = diskcache.Cache(str(directory))

    def get(self, key: str) -> dict[str, Any] | None:
        value: Any = self._cache.get(key)
        return None if value is None else cast(dict[str, Any], json.loads(str(value)))

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._cache.set(key, json.dumps(value, ensure_ascii=False))

    def __len__(self) -> int:
        return int(len(self._cache))

    def close(self) -> None:
        self._cache.close()
