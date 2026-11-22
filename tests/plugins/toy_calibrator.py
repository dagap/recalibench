"""Identity calibrator: returns its input. Exercises the Calibrator contract."""

import numpy as np

from recalibench.core.protocols import FloatArray, IntArray


class ToyIdentity:
    supports = frozenset({"binary", "multiclass"})

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, P: FloatArray, y: IntArray) -> None:
        return None

    def predict(self, P: FloatArray) -> FloatArray:
        return np.asarray(P, dtype=np.float64).copy()
