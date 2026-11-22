"""Evaluate stage: test-split metrics for every calibrated task x model -> metrics.json.

Raw scores and the selected calibrator get paired-bootstrap 95% CIs (and a CI on
their difference); every other candidate gets point estimates. Smooth ECE is a
point estimate (it is ~100x slower than binned ECE). Selective prediction and
reliability bins are stored too, so the report reads numbers, not models.

Refuses calibrators fitted on different scores (fingerprint guard).
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from recalibench.core.paths import Paths
from recalibench.core.protocols import FloatArray, IntArray
from recalibench.core.registry import Registry
from recalibench.evaluate import metrics as m
from recalibench.evaluate.bootstrap import paired_bootstrap
from recalibench.evaluate.selective import selective_summary
from recalibench.pipeline import tracking
from recalibench.pipeline.calibrate import CALIBRATED, label_indices, read_selected
from recalibench.pipeline.scores_io import Scores, read_meta, read_scores

log = logging.getLogger(__name__)

EVALUATE_VERSION = 2  # 2: reliability.bins
METRICS = "metrics.json"
TARGET_ACCURACY = 0.95


class StaleCalibration(Exception):
    """selected.json was fitted on different scores than the ones on disk."""


@dataclass
class EvalResult:
    task: str
    model: str
    status: str  # evaluated | fresh | skipped | failed
    reason: str = ""
    selected: str = ""
    raw_ece: float = float("nan")
    selected_ece: float = float("nan")


def metric_fns(kind: str, bins: int) -> dict[str, Callable[[FloatArray, IntArray], float]]:
    fns: dict[str, Callable[[FloatArray, IntArray], float]] = {
        "log_loss": m.log_loss,
        "brier": m.brier,
        "accuracy": m.accuracy,
        "ece": lambda P, y: m.top_label_ece(P, y, bins, "width"),
        "ece_equal_mass": lambda P, y: m.top_label_ece(P, y, bins, "mass"),
        "classwise_ece": lambda P, y: m.classwise_ece(P, y, bins),
    }
    if kind == "binary":
        fns["auroc"] = m.auroc
    return fns


def point_metrics(P: FloatArray, y: IntArray, kind: str, bins: int) -> dict[str, float]:
    out = {name: fn(P, y) for name, fn in metric_fns(kind, bins).items()}
    out["smooth_ece"] = m.smooth_ece(P, y)
    return out


def reliability_bins(P: FloatArray, y: IntArray, kind: str, bins: int) -> dict[str, Any]:
    conf, correct = P.max(axis=1), (P.argmax(axis=1) == y).astype(np.float64)
    out: dict[str, Any] = {"top_label": m.reliability(conf, correct, bins)}
    if kind == "binary":
        out["p_yes"] = m.reliability(P[:, 0], (y == 0).astype(np.float64), bins)
    return out


def calibrated_probs(directory: Path, scores: Scores) -> dict[str, FloatArray]:
    """Every calibrator's probabilities, rows aligned with scores.frame."""
    frame = pd.read_parquet(directory / CALIBRATED)
    cols = [f"p_{k}" for k in scores.keys]
    order = {i: n for n, i in enumerate(scores.frame["id"])}
    out: dict[str, FloatArray] = {}
    for name, part in frame.groupby("calibrator", sort=False):
        part = part.assign(_row=part["id"].map(order)).sort_values("_row")
        out[str(name)] = part[cols].to_numpy(dtype=np.float64)
    return out


def _fingerprint(selected: dict[str, Any], cfg: DictConfig) -> str:
    blob = json.dumps(
        {
            "v": EVALUATE_VERSION,
            "selected": selected["fingerprint"],
            "iters": int(cfg.bootstrap_iters),
            "seed": int(cfg.seed),
            "bins": int(cfg.ece_bins),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def evaluate_pair(directory: Path, cfg: DictConfig, force: bool = False) -> EvalResult:
    scores = read_scores(directory)
    meta = scores.meta
    task, model_id, kind = str(meta["task"]), str(meta["model_id"]), str(meta["kind"])
    selected = read_selected(directory)
    if selected is None:
        return EvalResult(task, model_id, "skipped", "not calibrated")
    if selected["scores_fingerprint"] != meta["fingerprint"]:
        raise StaleCalibration(
            f"{task} x {model_id}: calibrators were fitted on other scores; "
            "run `recal calibrate` again"
        )
    fp = _fingerprint(selected, cfg)
    path = directory / METRICS
    if not force and path.exists():
        old = cast(dict[str, Any], json.loads(path.read_text()))
        if old.get("fingerprint") == fp:
            log.info("%s x %s: metrics up to date, skipped", task, model_id)
            return EvalResult(task, model_id, "fresh", selected=str(selected["selected"]))

    bins, iters, seed = int(cfg.ece_bins), int(cfg.bootstrap_iters), int(cfg.seed)
    y_all = label_indices(scores)
    split = scores.frame["split"].to_numpy()
    test, val = split == "test", split == "val"
    raw = scores.probs()
    cal = calibrated_probs(directory, scores)
    winner = str(selected["selected"])
    y_test = y_all[test]

    boot = paired_bootstrap(
        metric_fns(kind, bins), raw[test], cal[winner][test], y_test, iters, seed
    )
    candidates = {name: point_metrics(P[test], y_test, kind, bins) for name, P in cal.items()}
    raw_point = point_metrics(raw[test], y_test, kind, bins)
    result: dict[str, Any] = {
        "task": task,
        "dataset": meta.get("dataset"),
        "model": meta.get("model"),
        "model_id": model_id,
        "kind": kind,
        "keys": scores.keys,
        "mode": meta.get("mode"),
        "n_test": int(test.sum()),
        "selected": winner,
        "fingerprint": fp,
        "scores_fingerprint": meta["fingerprint"],
        "ci": {"level": 0.95, "iters": iters, "method": "paired bootstrap over test rows"},
        "raw": {
            **{k: v["raw"].as_dict() for k, v in boot.items()},
            "smooth_ece": {"value": raw_point["smooth_ece"]},
        },
        "calibrated": {
            **{k: v["calibrated"].as_dict() for k, v in boot.items()},
            "smooth_ece": {"value": candidates[winner]["smooth_ece"]},
        },
        "difference": {k: v["difference"].as_dict() for k, v in boot.items()},
        "candidates": candidates,
        "reliability": {
            "bins": bins,  # equal-width bins over [0, 1]; only non-empty bins are listed
            "raw": reliability_bins(raw[test], y_test, kind, bins),
            "calibrated": reliability_bins(cal[winner][test], y_test, kind, bins),
        },
        "selective": {
            "raw": selective_summary(raw[val], y_all[val], raw[test], y_test, TARGET_ACCURACY),
            "calibrated": selective_summary(
                cal[winner][val], y_all[val], cal[winner][test], y_test, TARGET_ACCURACY
            ),
        },
        "cost_usd": meta.get("cost_usd"),
        "mean_latency_ms": meta.get("mean_latency_ms"),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(result, indent=2, default=float) + "\n")
    tracking.log_metrics(cfg, result)
    raw_ece, sel_ece = result["raw"]["ece"]["value"], result["calibrated"]["ece"]["value"]
    log.info("%s x %s: test ECE %.4f -> %.4f (%s)", task, model_id, raw_ece, sel_ece, winner)
    return EvalResult(
        task, model_id, "evaluated", selected=winner, raw_ece=raw_ece, selected_ece=sel_ece
    )


def evaluate(reg: Registry, paths: Paths, cfg: DictConfig, force: bool = False) -> list[EvalResult]:
    results: list[EvalResult] = []
    for task in [str(t) for t in cfg.experiment.tasks]:
        for name in [str(x) for x in cfg.experiment.models]:
            model = reg.model(name)
            directory = paths.artifact_dir(task, model.name, model.version)
            label = f"{model.name}@{model.version}"
            if read_meta(directory) is None:
                results.append(EvalResult(task, label, "skipped", "not scored"))
                continue
            results.append(evaluate_pair(directory, cfg, force))
    return results
