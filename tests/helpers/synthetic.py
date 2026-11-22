"""Synthetic scores with a known miscalibration, for calibrator tests.

Labels are drawn from the *true* probabilities, so the true probabilities are
calibrated by construction; the returned scores are a known distortion of them.
y follows the calibrator contract: the column index of the true class
(binary: column 0 = yes).
"""

import numpy as np

from recalibench.core.protocols import FloatArray, IntArray


def _softmax(Z: FloatArray) -> FloatArray:
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def binary(
    n: int, slope: float = 1.0, bias: float = 0.0, seed: int = 0
) -> tuple[FloatArray, IntArray]:
    """Scores sigmoid(slope * z + bias) for true logit z; slope < 1 under-, > 1 over-confident."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, n)
    yes = rng.random(n) < 1 / (1 + np.exp(-z))
    p = 1 / (1 + np.exp(-(slope * z + bias)))
    return np.column_stack([p, 1 - p]), np.where(yes, 0, 1).astype(np.int64)


def multiclass(
    n: int,
    k: int = 5,
    temperature: float = 1.0,
    class_bias: FloatArray | None = None,
    seed: int = 0,
) -> tuple[FloatArray, IntArray]:
    """Scores softmax(Z / temperature + class_bias) for true logits Z."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(0.0, 1.5, (n, k))
    true = _softmax(Z)
    y = np.array([rng.choice(k, p=row) for row in true], dtype=np.int64)
    shift = np.zeros(k) if class_bias is None else class_bias
    return _softmax(Z / temperature + shift), y
