"""A minimal, independent binned ECE: evaluate/metrics.py is cross-checked against it."""

import numpy as np
import numpy.typing as npt


def ece(confidence: npt.ArrayLike, correct: npt.ArrayLike, bins: int = 15) -> float:
    """Equal-width expected calibration error: sum_b |acc_b - conf_b| * n_b / n."""
    conf = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    idx = np.minimum((conf * bins).astype(np.int64), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += abs(float(hit[mask].mean()) - float(conf[mask].mean())) * float(mask.mean())
    return total


def binary_ece(p_yes: npt.ArrayLike, labels: npt.ArrayLike, bins: int = 15) -> float:
    """Calibration of P(yes) against the 0/1 label, binned on P(yes)."""
    return ece(p_yes, labels, bins)
