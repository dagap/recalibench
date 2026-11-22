"""Calibration and accuracy metrics on a probability matrix.

Conventions (shared with calibrators): P is (n, k) with columns in the scores'
key order, y holds the column index of the true class. Binary tasks have
k = 2 columns [yes, no].

Binned ECE (equal-width and equal-mass, top-label and class-wise) is the
familiar number; smooth ECE (Blasiok & Nakkiran 2023, via relplot) needs no
bins and is steadier on small test sets. Cross-checked against scikit-learn in
the tests.
"""

from typing import Any, cast

import numpy as np
import relplot  # pyright: ignore[reportMissingTypeStubs]

from recalibench.core.protocols import FloatArray, IntArray

CLIP = 1e-15


def log_loss(P: FloatArray, y: IntArray) -> float:
    """Mean negative log-probability of the true class."""
    p = np.clip(P[np.arange(len(y)), y], CLIP, 1.0)
    return float(-np.log(p).mean())


def brier(P: FloatArray, y: IntArray) -> float:
    """Binary (k = 2): mean (p_yes - [y is yes])^2. Multiclass: mean sum_k (P - onehot)^2."""
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    total = float(((P - onehot) ** 2).sum(axis=1).mean())
    return total / 2 if P.shape[1] == 2 else total


def _bin_index(conf: FloatArray, bins: int, scheme: str) -> IntArray:
    if scheme == "width":
        return np.minimum((conf * bins).astype(np.int64), bins - 1)
    if scheme == "mass":  # equal-count bins: rank-based so ties split evenly
        order = np.argsort(conf, kind="stable")
        idx = np.empty(len(conf), dtype=np.int64)
        idx[order] = np.arange(len(conf)) * bins // max(1, len(conf))
        return idx
    raise ValueError(f"unknown bin scheme {scheme!r}")


def binned_ece(
    conf: FloatArray, outcome: FloatArray, bins: int = 15, scheme: str = "width"
) -> float:
    """sum_b |mean(outcome_b) - mean(conf_b)| * n_b / n."""
    idx = _bin_index(conf, bins, scheme)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            gap = abs(float(outcome[mask].mean()) - float(conf[mask].mean()))
            total += gap * float(mask.mean())
    return total


def _top(P: FloatArray, y: IntArray) -> tuple[FloatArray, FloatArray]:
    return P.max(axis=1), (P.argmax(axis=1) == y).astype(np.float64)


def top_label_ece(P: FloatArray, y: IntArray, bins: int = 15, scheme: str = "width") -> float:
    """ECE of the top-label confidence against whether the top label is right."""
    return binned_ece(*_top(P, y), bins=bins, scheme=scheme)


def classwise_ece(P: FloatArray, y: IntArray, bins: int = 15) -> float:
    """Mean over classes of the equal-width ECE of P[:, k] against [y == k].

    For a binary [yes, no] matrix this equals the usual ECE of P(yes)."""
    k = P.shape[1]
    cols = [binned_ece(P[:, c], (y == c).astype(np.float64), bins) for c in range(k)]
    return float(np.mean(cols[:1] if k == 2 else cols))


def smooth_ece(P: FloatArray, y: IntArray) -> float:
    """Smooth ECE: P(yes) for binary, top-label confidence for multiclass."""
    if P.shape[1] == 2:
        f, outcome = P[:, 0], (y == 0).astype(np.float64)
    else:
        f, outcome = _top(P, y)
    value = cast(Any, relplot.smECE(f, outcome))  # pyright: ignore[reportUnknownMemberType]
    return float(value)


def auroc(P: FloatArray, y: IntArray) -> float:
    """Binary AUROC of P(yes) (column 0), Mann-Whitney with average ranks for ties."""
    score, pos = P[:, 0], y == 0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def reliability(
    conf: FloatArray, outcome: FloatArray, bins: int = 15, scheme: str = "width"
) -> list[dict[str, float]]:
    """Per non-empty bin: mean confidence, observed rate, count (for reliability diagrams)."""
    idx = _bin_index(conf, bins, scheme)
    out: list[dict[str, float]] = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append(
                {
                    "bin": b,
                    "confidence": float(conf[mask].mean()),
                    "observed": float(outcome[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return out


def accuracy(P: FloatArray, y: IntArray) -> float:
    return float((P.argmax(axis=1) == y).mean())
