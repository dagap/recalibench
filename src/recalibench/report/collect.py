"""Read everything the report shows from artifacts*/: metrics.json per task x model and
_experiments/<name>/summary.json. Nothing here knows a task, model or experiment by name.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from recalibench.core.paths import Paths

METRICS_FILE = "metrics.json"
SUMMARY_FILE = "summary.json"


class ReportError(Exception):
    """Nothing to report."""


@dataclass(frozen=True)
class Pair:
    task: str
    model_id: str
    kind: str
    dataset: str
    selected: str
    n_test: int
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ReportData:
    mode: str
    pairs: list[Pair]
    not_evaluated: list[tuple[str, str]]
    experiments: dict[str, dict[str, Any]]
    tasks: list[str]
    models: list[str]


def ci(metric: dict[str, Any] | None) -> tuple[float, float | None, float | None] | None:
    """{"value", "low", "high"} -> (value, low, high); a point estimate has no bounds."""
    if metric is None:
        return None
    low, high = metric.get("low"), metric.get("high")
    return (
        float(metric["value"]),
        None if low is None else float(low),
        None if high is None else float(high),
    )


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _pair(m: dict[str, Any]) -> Pair:
    return Pair(
        task=str(m["task"]),
        model_id=str(m["model_id"]),
        kind=str(m["kind"]),
        dataset=str(m.get("dataset", "")),
        selected=str(m["selected"]),
        n_test=int(m["n_test"]),
        metrics=m,
    )


def _subdirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


def collect(paths: Paths) -> ReportData:
    root = paths.artifacts_root()
    pairs: list[Pair] = []
    not_evaluated: list[tuple[str, str]] = []
    for task_dir in _subdirs(root):
        if task_dir.name.startswith("_"):
            continue
        for model_dir in _subdirs(task_dir):
            f = model_dir / METRICS_FILE
            if f.is_file():
                pairs.append(_pair(_read(f)))
            else:
                not_evaluated.append((task_dir.name, model_dir.name))
    if not pairs:
        raise ReportError(f"no {METRICS_FILE} under {root}; run `recal evaluate` first")
    pairs.sort(key=lambda p: (p.task, p.model_id))
    experiments = {
        d.name: _read(d / SUMMARY_FILE)
        for d in _subdirs(paths.experiments_root())
        if (d / SUMMARY_FILE).is_file()
    }
    return ReportData(
        mode=paths.mode,
        pairs=pairs,
        not_evaluated=not_evaluated,
        experiments=experiments,
        tasks=sorted({p.task for p in pairs}),
        models=sorted({p.model_id for p in pairs}),
    )
