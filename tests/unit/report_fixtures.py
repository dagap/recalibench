"""Synthetic metrics.json / summary.json files, shaped like pipeline/evaluate.py output."""

import json
from pathlib import Path
from typing import Any

METRIC_NAMES = ["log_loss", "brier", "accuracy", "ece", "ece_equal_mass", "classwise_ece"]


def _ci(value: float, low: float | None = None, high: float | None = None) -> dict[str, float]:
    return {
        "value": value,
        "low": value * 0.8 if low is None else low,
        "high": value * 1.2 if high is None else high,
    }


def _bins(n: int, shift: float, start: int = 0) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for b in range(start, n):
        conf = (b + 0.5) / n
        out.append(
            {"bin": b, "confidence": conf, "observed": min(1.0, conf + shift), "count": 10 + b}
        )
    return out


def _selective(aurc: float, coverage: float) -> dict[str, Any]:
    cov = [i / 49 for i in range(50)]
    return {
        "target_accuracy": 0.95,
        "aurc": aurc,
        "tuned": {"threshold": 0.7, "coverage": coverage, "accuracy": 0.94},
        "trust": {"threshold": 0.95, "coverage": coverage / 2, "accuracy": 0.99},
        "oracle_coverage": coverage,
        "curve": {"coverage": cov, "risk": [0.2 * c for c in cov]},
    }


def metrics_dict(
    *,
    task: str,
    model_id: str,
    kind: str = "binary",
    raw_ece: float = 0.10,
    cal_ece: float = 0.02,
    cal_ece_ci: tuple[float, float] | None = None,
    cost_usd: float = 0.0,
    mean_latency_ms: float = 0.0,
) -> dict[str, Any]:
    names = METRIC_NAMES + (["auroc"] if kind == "binary" else [])
    raw = {n: _ci(0.3) for n in names}
    cal = {n: _ci(0.25) for n in names}
    raw["ece"] = _ci(raw_ece)
    cal["ece"] = _ci(cal_ece, *(cal_ece_ci or (None, None)))
    raw["smooth_ece"] = {"value": raw_ece * 1.1}
    cal["smooth_ece"] = {"value": cal_ece * 1.1}
    diff = {n: _ci(cal[n]["value"] - raw[n]["value"], -0.1, 0.1) for n in names}
    rel_raw: dict[str, Any] = {"top_label": _bins(15, -0.1, start=7)}
    rel_cal: dict[str, Any] = {"top_label": _bins(15, 0.0, start=7)}
    if kind == "binary":
        rel_raw["p_yes"] = _bins(15, -0.1)
        rel_cal["p_yes"] = _bins(15, 0.0)
    return {
        "task": task,
        "dataset": "toy",
        "model": model_id.split("@")[0],
        "model_id": model_id,
        "kind": kind,
        "keys": ["yes", "no"] if kind == "binary" else ["a", "b", "c"],
        "mode": "mock",
        "n_test": 600,
        "selected": "platt",
        "fingerprint": "f" * 64,
        "scores_fingerprint": "s" * 64,
        "ci": {"level": 0.95, "iters": 1000, "method": "paired bootstrap over test rows"},
        "raw": raw,
        "calibrated": cal,
        "difference": diff,
        "candidates": {"platt": {n: cal[n]["value"] for n in cal}},
        "reliability": {"bins": 15, "raw": rel_raw, "calibrated": rel_cal},
        "selective": {"raw": _selective(0.06, 0.5), "calibrated": _selective(0.04, 0.7)},
        "cost_usd": cost_usd,
        "mean_latency_ms": mean_latency_ms,
        "written_at": "2026-11-22T16:57:58+00:00",
    }


def write_metrics(arts: Path, **kwargs: Any) -> Path:
    """Write <arts>/<task>/<model_id>/metrics.json; kwargs as metrics_dict()."""
    m = metrics_dict(**kwargs)
    d = arts / m["task"] / m["model_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(m))
    return d


def write_summary(arts: Path, name: str, summary: dict[str, Any]) -> None:
    d = arts / "_experiments" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary))


def wording_summary() -> dict[str, Any]:
    variants = [
        {
            "variant": f"phrasing_{i}",
            "question": f"Question {i}?",
            "selected": "beta",
            "raw_ece": _ci(0.1 + 0.01 * i),
            "calibrated_ece": _ci(0.0163, 0.0185, 0.0518),
            "raw_log_loss": _ci(0.4),
            "calibrated_log_loss": _ci(0.3),
        }
        for i in range(3)
    ]
    return {"variants": variants, "raw_ece_spread": 0.028, "calibrated_ece_spread": 0.009}


def option_order_summary() -> dict[str, Any]:
    variants = [
        {
            "variant": f"order_{i}",
            "order": ["a", "b", "c"][i:] + ["a", "b", "c"][:i],
            "first_option": "abc"[i],
            "first_option_true_share": 0.05,
            "selected": "vector_scaling",
            "raw_first_option_mean_prob": 0.15,
            "raw_first_option_predicted_share": 0.16,
            "raw_classwise_ece": 0.12,
            "calibrated_first_option_mean_prob": 0.05,
            "calibrated_first_option_predicted_share": 0.0,
            "calibrated_classwise_ece": 0.03,
        }
        for i in range(3)
    ]
    return {
        "variants": variants,
        "raw_mean_prob_instability": 0.04,
        "calibrated_mean_prob_instability": 0.0004,
    }


def transfer_summary() -> dict[str, Any]:
    row = {n: 0.2 for n in METRIC_NAMES}
    return {
        "from": "toy_binary",
        "to": "toy_binary_text",
        "model_id": "mock@mock-1",
        "transferred_calibrator": "beta",
        "native_calibrator": "platt",
        "note": "deliberately applies a calibrator to scores it was not fitted on",
        "raw": row,
        "transferred": row,
        "native": row,
    }
