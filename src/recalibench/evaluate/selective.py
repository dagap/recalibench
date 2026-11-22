"""Selective prediction: handle the most confident cases automatically, send the rest on.

risk-coverage  sort by confidence (most sure first); at coverage c, risk is the error
               rate among the top c of cases
AURC           mean risk over all coverage levels (lower is better)
auto-handle    the share of cases handled automatically while staying >= target accuracy,
               under two policies:
                 tuned   threshold chosen on val, then applied to test
                 trust   auto-handle wherever confidence >= target (no tuning): only works
                         if the confidence numbers mean what they say
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from recalibench.core.protocols import FloatArray, IntArray


def confidence_and_correct(P: FloatArray, y: IntArray) -> tuple[FloatArray, FloatArray]:
    return P.max(axis=1), (P.argmax(axis=1) == y).astype(np.float64)


def risk_coverage(conf: FloatArray, correct: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(-conf, kind="stable")
    errors = np.cumsum(1.0 - correct[order])
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), errors / k


def aurc(conf: FloatArray, correct: FloatArray) -> float:
    return float(risk_coverage(conf, correct)[1].mean())


def threshold_for_accuracy(conf: FloatArray, correct: FloatArray, target: float) -> float | None:
    """Lowest confidence threshold whose auto-handled set still reaches `target` accuracy."""
    order = np.argsort(-conf, kind="stable")
    acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(acc >= target)
    if len(ok) == 0:
        return None
    return float(conf[order][ok[-1]])


@dataclass(frozen=True)
class Policy:
    threshold: float | None
    coverage: float
    accuracy: float | None

    def as_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "coverage": self.coverage, "accuracy": self.accuracy}


def apply_threshold(conf: FloatArray, correct: FloatArray, threshold: float | None) -> Policy:
    if threshold is None:
        return Policy(None, 0.0, None)
    mask = conf >= threshold
    accuracy = float(correct[mask].mean()) if mask.any() else None
    return Policy(threshold, float(mask.mean()), accuracy)


def selective_summary(
    P_val: FloatArray,
    y_val: IntArray,
    P_test: FloatArray,
    y_test: IntArray,
    target: float = 0.95,
    curve_points: int = 50,
) -> dict[str, Any]:
    conf_v, ok_v = confidence_and_correct(P_val, y_val)
    conf_t, ok_t = confidence_and_correct(P_test, y_test)
    cov, risk = risk_coverage(conf_t, ok_t)
    pick = np.unique(np.linspace(0, len(cov) - 1, min(curve_points, len(cov))).astype(int))
    return {
        "target_accuracy": target,
        "aurc": aurc(conf_t, ok_t),
        "tuned": apply_threshold(
            conf_t, ok_t, threshold_for_accuracy(conf_v, ok_v, target)
        ).as_dict(),
        "trust": apply_threshold(conf_t, ok_t, target).as_dict(),
        "oracle_coverage": apply_threshold(
            conf_t, ok_t, threshold_for_accuracy(conf_t, ok_t, target)
        ).coverage,
        "curve": {"coverage": cov[pick].tolist(), "risk": risk[pick].tolist()},
    }
