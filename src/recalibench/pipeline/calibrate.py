"""Calibrate stage: for every scored task x model, fit all compatible calibrators
on the fit split, choose one on val, and save

    artifacts*/<task>/<model>@<version>/calibrators/<name>.pkl
    artifacts*/<task>/<model>@<version>/selected.json
    artifacts*/<task>/<model>@<version>/calibrated.parquet   (id, split, calibrator, p_<key>...)

Calibrators are applied to every split (test included) but fitted and chosen
on fit and val only. selected.json carries the scores' fingerprint, so later
stages can refuse calibrators fitted on different scores (or a different version).
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from recalibench.calibrators.select import RULE, choose, fit_candidates, val_metrics
from recalibench.core.paths import Paths
from recalibench.core.protocols import Calibrator, IntArray
from recalibench.core.registry import Registry
from recalibench.pipeline.scores_io import Scores, read_meta, read_scores

log = logging.getLogger(__name__)

CALIBRATE_VERSION = 1
SELECTED = "selected.json"
CALIBRATED = "calibrated.parquet"


@dataclass
class CalibrationResult:
    task: str
    model: str
    status: str  # calibrated | fresh | skipped | failed
    reason: str = ""
    selected: str = ""
    raw_val_log_loss: float = float("nan")
    selected_val_log_loss: float = float("nan")


def label_indices(scores: Scores) -> IntArray:
    """Column index of each row's true class (binary: 1 -> 'yes' = column 0)."""
    keys = scores.keys
    labels = scores.labels()
    if scores.meta["kind"] == "binary":
        names = ["yes" if int(v) == 1 else "no" for v in labels]
    else:
        names = [str(v) for v in labels]
    unknown = set(names) - set(keys)
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} are not score columns {keys}")
    return np.array([keys.index(n) for n in names], dtype=np.int64)


def calibrator_names(reg: Registry, cfg: DictConfig, kind: str) -> list[str]:
    wanted = cfg.experiment.calibrators
    names = reg.names("calibrator") if str(wanted) == "auto" else [str(n) for n in wanted]
    need = "binary" if kind == "binary" else "multiclass"
    out: list[str] = []
    for n in names:
        cal = reg.calibrator(n)
        if need in cal.supports:
            out.append(n)
        else:
            log.debug("calibrator %s skipped: supports %s, need %s", n, sorted(cal.supports), need)
    return out


def _fingerprint(reg: Registry, scores_fp: str, names: list[str]) -> str:
    configs = {n: OmegaConf.to_container(reg.config("calibrator", n), resolve=True) for n in names}
    blob = json.dumps(
        {"v": CALIBRATE_VERSION, "scores": scores_fp, "calibrators": configs},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def read_selected(directory: Path) -> dict[str, Any] | None:
    path = directory / SELECTED
    return cast(dict[str, Any], json.loads(path.read_text())) if path.exists() else None


def calibrate_pair(
    reg: Registry, directory: Path, cfg: DictConfig, force: bool = False
) -> CalibrationResult:
    scores = read_scores(directory)
    meta = scores.meta
    task, model_id, kind = str(meta["task"]), str(meta["model_id"]), str(meta["kind"])
    names = calibrator_names(reg, cfg, kind)
    fp = _fingerprint(reg, str(meta["fingerprint"]), names)
    old = read_selected(directory)
    if not force and old is not None and old.get("fingerprint") == fp:
        log.info("%s x %s: calibrators up to date, skipped", task, model_id)
        return CalibrationResult(task, model_id, "fresh", selected=str(old["selected"]))

    y = label_indices(scores)
    split = scores.frame["split"].to_numpy()
    P = scores.probs()
    fit, val = split == "fit", split == "val"
    if not fit.any() or not val.any():
        return CalibrationResult(task, model_id, "failed", "needs non-empty fit and val splits")

    bins = int(cfg.ece_bins)
    cals: dict[str, Calibrator] = {n: reg.calibrator(n) for n in names}
    candidates = fit_candidates(cals, P[fit], y[fit], P[val], y[val], bins)
    for c in candidates:
        if c.error:
            log.warning("%s x %s: calibrator %s failed: %s", task, model_id, c.name, c.error)
    try:
        winner = choose(candidates)
    except ValueError as e:
        return CalibrationResult(task, model_id, "failed", str(e))

    cal_dir = directory / "calibrators"
    cal_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    keys = scores.keys
    for c in candidates:
        if c.error:
            continue
        with (cal_dir / f"{c.name}.pkl").open("wb") as f:
            pickle.dump(c.calibrator, f)
        Q = c.calibrator.predict(P)
        part = pd.DataFrame(Q, columns=[f"p_{k}" for k in keys])
        part.insert(0, "calibrator", c.name)
        part.insert(0, "split", split)
        part.insert(0, "id", scores.frame["id"].to_numpy())
        frames.append(part)
    pd.concat(frames, ignore_index=True).to_parquet(directory / CALIBRATED, index=False)

    raw = val_metrics(P[val], y[val], bins)
    selected: dict[str, Any] = {
        "task": task,
        "model_id": model_id,
        "kind": kind,
        "keys": keys,
        "scores_fingerprint": meta["fingerprint"],
        "fingerprint": fp,
        "selected": winner.name,
        "rule": RULE,
        "raw_val": raw,
        "candidates": {c.name: c.summary() for c in candidates},
        "n_fit": int(fit.sum()),
        "n_val": int(val.sum()),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (directory / SELECTED).write_text(json.dumps(selected, indent=2, default=float) + "\n")
    log.info(
        "%s x %s: selected %s (val log loss %.4f -> %.4f, ECE %.4f -> %.4f)",
        task,
        model_id,
        winner.name,
        raw["log_loss"],
        winner.val_log_loss,
        raw["ece"],
        winner.val_ece,
    )
    return CalibrationResult(
        task,
        model_id,
        "calibrated",
        selected=winner.name,
        raw_val_log_loss=raw["log_loss"],
        selected_val_log_loss=winner.val_log_loss,
    )


def calibrate(
    reg: Registry, paths: Paths, cfg: DictConfig, force: bool = False
) -> list[CalibrationResult]:
    results: list[CalibrationResult] = []
    for task in [str(t) for t in cfg.experiment.tasks]:
        for name in [str(m) for m in cfg.experiment.models]:
            model = reg.model(name)
            directory = paths.artifact_dir(task, model.name, model.version)
            label = f"{model.name}@{model.version}"
            if read_meta(directory) is None:
                results.append(CalibrationResult(task, label, "skipped", "not scored"))
                continue
            results.append(calibrate_pair(reg, directory, cfg, force))
    return results
