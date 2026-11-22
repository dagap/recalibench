"""MANIFEST.json for downloaded raw data: where it came from and its checksum."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

MANIFEST = "MANIFEST.json"


class DownloadError(Exception):
    """A dataset could not be fetched. The message tells the user how to fix it."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(raw_dir: Path, file: Path, source: str, **extra: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "source": source,
        "file": file.name,
        "sha256": sha256_file(file),
        "bytes": file.stat().st_size,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    (raw_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(raw_dir: Path) -> dict[str, Any] | None:
    path = raw_dir / MANIFEST
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text()))


def is_current(raw_dir: Path, file: Path, expected_sha256: str | None = None) -> bool:
    """True if `file` exists and matches its manifest (and the pinned checksum, if any)."""
    manifest = read_manifest(raw_dir)
    if manifest is None or not file.exists() or manifest.get("file") != file.name:
        return False
    actual = sha256_file(file)
    if expected_sha256 is not None and actual != expected_sha256:
        return False
    return actual == manifest.get("sha256")
