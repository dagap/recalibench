"""Prepare stage: build each experiment task's processed items, skipping fresh ones."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.core.types import TaskSpec
from recalibench.datasets.manifest import read_manifest
from recalibench.tasks.builder import (
    RATE_TOLERANCE,
    BuildError,
    build_frame,
    check_splits,
    rate_spread,
    write,
)

log = logging.getLogger(__name__)

BUILDER_VERSION = 1  # bump when builder output changes for the same inputs


class PrepareError(Exception):
    """A task can't be prepared yet (e.g. its raw data isn't downloaded)."""


@dataclass(frozen=True)
class PrepareResult:
    task: str
    skipped: bool
    meta: dict[str, Any]


def _dataset_fingerprint(raw_dir: Path) -> Any:
    manifest = read_manifest(raw_dir)
    if manifest is not None:
        return manifest["sha256"]
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    return [(str(p.relative_to(raw_dir)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]


def task_settings(spec: TaskSpec, cfg: DictConfig) -> tuple[int, dict[str, float]]:
    sample_size = spec.sample_size if spec.sample_size is not None else int(cfg.sample_size)
    raw = spec.splits if spec.splits is not None else OmegaConf.to_container(cfg.splits)
    fractions = {str(k): float(v) for k, v in cast(dict[str, Any], raw).items()}
    return sample_size, fractions


def fingerprint(spec: TaskSpec, cfg: DictConfig, raw_dir: Path) -> str:
    sample_size, fractions = task_settings(spec, cfg)
    payload = {
        "builder": BUILDER_VERSION,
        "dataset": spec.dataset,
        "question": [spec.question.kind, spec.question.text, spec.question.keys],
        "state": spec.state,
        "label": spec.label,
        "seed": int(cfg.seed),
        "sample_size": sample_size,
        "splits": fractions,
        "raw": _dataset_fingerprint(raw_dir),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def prepare_task(
    reg: Registry, paths: Paths, cfg: DictConfig, name: str, force: bool = False
) -> PrepareResult:
    spec = reg.task(name)
    raw_dir = paths.raw_dir(spec.dataset)
    if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
        raise PrepareError(
            f"{name}: no raw data for dataset {spec.dataset!r} in {raw_dir}; "
            f"run `recal download` first"
        )
    fp = fingerprint(spec, cfg, raw_dir)
    meta_path, out = paths.processed_meta(name), paths.processed(name)
    if not force and meta_path.exists() and out.exists():
        old = cast(dict[str, Any], json.loads(meta_path.read_text()))
        if old.get("fingerprint") == fp:
            log.info("%s: up to date, skipped", name)
            return PrepareResult(name, skipped=True, meta=old)

    dataset = reg.dataset(spec.dataset)
    sample_size, fractions = task_settings(spec, cfg)
    built = build_frame(
        spec,
        dataset.load(raw_dir),
        seed=int(cfg.seed),
        sample_size=sample_size,
        fractions=fractions,
    )
    problems = check_splits(built.frame, spec, list(fractions))
    if problems:
        raise BuildError(f"{name}: " + "; ".join(problems))
    if len(built.frame) < sample_size:
        log.warning(
            "%s: only %d rows available (sample_size %d)", name, len(built.frame), sample_size
        )
    spread = rate_spread(built.frame)
    worst = max(spread.values(), default=0.0)
    if worst > RATE_TOLERANCE:
        log.warning("%s: label rates differ by %.3f across splits", name, worst)
    meta = write(built, spec, paths, fp, {"max_rate_spread": round(worst, 6)})
    counts = " ".join(f"{s}={meta['counts'].get(s, 0)}" for s in fractions)
    rates = meta["label_rates"].get(next(iter(fractions)), {})
    log.info("%s: %d rows (%s); fit label shares %s", name, meta["rows"], counts, rates)
    return PrepareResult(name, skipped=False, meta=meta)


def prepare(
    reg: Registry, paths: Paths, cfg: DictConfig, force: bool = False
) -> list[PrepareResult]:
    return [prepare_task(reg, paths, cfg, str(t), force) for t in cfg.experiment.tasks]
