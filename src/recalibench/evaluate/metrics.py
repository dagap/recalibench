"""Calibration and accuracy metrics on a probability matrix.

Conventions (shared with calibrators): P is (n, k) with columns in the scores'
key order, y holds the column index of the true class. Binary tasks have
k = 2 columns [yes, no].

Phase 6 needs log loss, Brier and top-label ECE for selection; Phase 7 adds
the rest (class-wise ECE, equal-mass bins, AUROC, bootstrap, netcal check).
"""

import numpy as np

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


def top_label_ece(P: FloatArray, y: IntArray, bins: int = 15) -> float:
    """Equal-width ECE of the top-label confidence against whether the top label is right."""
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(np.float64)
    idx = np.minimum((conf * bins).astype(np.int64), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += abs(float(correct[mask].mean()) - float(conf[mask].mean())) * float(
                mask.mean()
            )
    return total


def accuracy(P: FloatArray, y: IntArray) -> float:
    return float((P.argmax(axis=1) == y).mean())
