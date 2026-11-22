"""The only place that builds data, cache, artifact and report paths.

Mock and live runs never share a cache or artifact directory: a mock response
served from a live cache would masquerade as a real model answer.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig

Mode = Literal["mock", "live"]
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _part(value: str, what: str) -> str:
    if not _SAFE.match(value) or value in {".", ".."}:
        raise ValueError(f"{what} {value!r} is not a safe path component ([A-Za-z0-9_.-]+)")
    return value


def model_key(model: str, version: str) -> str:
    """`name@version`, the identity used in cache keys and artifact paths."""
    return f"{_part(model, 'model name')}@{_part(version, 'model version')}"


@dataclass(frozen=True)
class Paths:
    root: Path
    mode: Mode

    def __post_init__(self) -> None:
        if self.mode not in ("mock", "live"):
            raise ValueError(f"mode must be 'mock' or 'live', got {self.mode!r}")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Paths":
        return cls(root=Path(str(cfg.paths.root)).resolve(), mode=cfg.mode)

    @property
    def _suffix(self) -> str:
        return "_mock" if self.mode == "mock" else ""

    def raw_dir(self, dataset: str) -> Path:
        return self.root / "data" / "raw" / _part(dataset, "dataset")

    def processed(self, task: str) -> Path:
        return self.root / "data" / "processed" / f"{_part(task, 'task')}.parquet"

    def cache_dir(self, model: str, version: str) -> Path:
        return self.root / f"cache{self._suffix}" / model_key(model, version)

    def artifact_dir(self, task: str, model: str, version: str) -> Path:
        base = self.root / f"artifacts{self._suffix}"
        return base / _part(task, "task") / model_key(model, version)

    def reports_dir(self) -> Path:
        return self.root / "reports"
