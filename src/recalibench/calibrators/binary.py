"""Calibrators for binary tasks, plus identity and temperature (which serve both kinds).

Data contract (all calibrators): fit(P, y) / predict(P) on the full (n, k)
probability matrix, columns in the scores' key order; y is the column index of
the true class. Binary tasks have k = 2 columns [yes, no], so "p" below is
P[:, 0] and "positive" means y == 0.

`parametric` marks families selection prefers on a near-tie (select.py).
"""

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression  # pyright: ignore[reportMissingTypeStubs]

from recalibench.calibrators._fit import (
    CLIP,
    binary_logistic,
    finish,
    fit_temperature,
    safe_log,
    softmax,
)
from recalibench.core.protocols import FloatArray, IntArray


def _p(P: FloatArray) -> FloatArray:
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"binary calibrators need an (n, 2) [yes, no] matrix, got {P.shape}")
    return np.clip(P[:, 0], CLIP, 1 - CLIP)


def _pair(q: FloatArray) -> FloatArray:
    return finish(np.column_stack([q, 1.0 - q]))


def _logit(p: FloatArray) -> FloatArray:
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-z))


class Identity:
    """No change: the raw scores, kept as a candidate so 'already calibrated' can win."""

    supports = frozenset({"binary", "multiclass"})
    parametric = True

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, P: FloatArray, y: IntArray) -> None:
        return None

    def predict(self, P: FloatArray) -> FloatArray:
        return finish(np.asarray(P, dtype=np.float64))

    def params(self) -> dict[str, Any]:
        return {}


class Temperature:
    """softmax(log P / T): one number that stretches (T < 1) or shrinks (T > 1) confidence.

    For a binary [yes, no] pair this is exactly sigmoid(logit(p) / T).
    """

    supports = frozenset({"binary", "multiclass"})
    parametric = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.T = 1.0

    def fit(self, P: FloatArray, y: IntArray) -> None:
        self.T = fit_temperature(safe_log(P), y)

    def predict(self, P: FloatArray) -> FloatArray:
        return finish(softmax(safe_log(P) / self.T))

    def params(self) -> dict[str, Any]:
        return {"T": self.T}


class Platt:
    """sigmoid(a * logit(p) + b): temperature plus a yes/no bias."""

    supports = frozenset({"binary"})
    parametric = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.a, self.b = 1.0, 0.0

    def fit(self, P: FloatArray, y: IntArray) -> None:
        w, self.b = binary_logistic(_logit(_p(P))[:, None], y)
        self.a = float(w[0])

    def predict(self, P: FloatArray) -> FloatArray:
        return _pair(_sigmoid(self.a * _logit(_p(P)) + self.b))

    def params(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b}


class Beta:
    """Beta calibration (Kull et al. 2017): logit q = a ln p - b ln(1 - p) + c, a, b >= 0.

    Treats the low and high ends separately. If a fitted coefficient is negative
    the map would not be monotone, so that feature is dropped and the fit redone.
    """

    supports = frozenset({"binary"})
    parametric = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.a, self.b, self.c = 1.0, 1.0, 0.0

    @staticmethod
    def _features(p: FloatArray) -> FloatArray:
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, P: FloatArray, y: IntArray) -> None:
        X = self._features(_p(P))
        w, c = binary_logistic(X, y)
        if w[0] < 0 or w[1] < 0:
            keep = 1 if w[0] < 0 else 0
            w1, c = binary_logistic(X[:, [keep]], y)
            w = np.zeros(2)
            w[keep] = max(0.0, float(w1[0]))
        self.a, self.b, self.c = float(w[0]), float(w[1]), float(c)

    def predict(self, P: FloatArray) -> FloatArray:
        X = self._features(_p(P))
        return _pair(_sigmoid(self.a * X[:, 0] + self.b * X[:, 1] + self.c))

    def params(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "c": self.c}


class Isotonic:
    """Monotone step function from p to the observed rate (sklearn IsotonicRegression)."""

    supports = frozenset({"binary"})
    parametric = False

    def __init__(self, name: str) -> None:
        self.name = name
        self._iso: Any = None

    def fit(self, P: FloatArray, y: IntArray) -> None:
        iso: Any = IsotonicRegression(y_min=CLIP, y_max=1 - CLIP, out_of_bounds="clip")
        iso.fit(_p(P), (y == 0).astype(np.float64))
        self._iso = iso

    def predict(self, P: FloatArray) -> FloatArray:
        if self._iso is None:
            raise RuntimeError("Isotonic.predict before fit")
        return _pair(np.asarray(self._iso.predict(_p(P)), dtype=np.float64))

    def params(self) -> dict[str, Any]:
        steps = 0 if self._iso is None else len(self._iso.X_thresholds_)
        return {"steps": int(steps)}


class Histogram:
    """Equal-mass bins on p; each bin outputs its observed rate, smoothed toward the base rate."""

    supports = frozenset({"binary"})
    parametric = False

    def __init__(self, name: str, n_bins: int = 15, prior_strength: float = 2.0) -> None:
        self.name = name
        self.n_bins = n_bins
        self.prior_strength = prior_strength
        self.edges: FloatArray = np.array([])
        self.values: FloatArray = np.array([])

    def fit(self, P: FloatArray, y: IntArray) -> None:
        p, t = _p(P), (y == 0).astype(np.float64)
        qs = np.quantile(p, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        self.edges = np.unique(qs)
        idx = np.searchsorted(self.edges, p, side="right")
        prior, m = float(t.mean()), self.prior_strength
        self.values = np.array(
            [
                (t[idx == b].sum() + m * prior) / ((idx == b).sum() + m)
                for b in range(len(self.edges) + 1)
            ]
        )

    def predict(self, P: FloatArray) -> FloatArray:
        return _pair(self.values[np.searchsorted(self.edges, _p(P), side="right")])

    def params(self) -> dict[str, Any]:
        return {"bins": len(self.values)}
