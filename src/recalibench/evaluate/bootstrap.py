"""Paired bootstrap over test rows.

Each iteration draws one set of row indices and scores raw and calibrated
probabilities on the *same* rows, so the CI of the difference reflects
sampling noise honestly. Calibrators are not refitted inside the bootstrap:
the CIs cover test-set sampling noise only.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from recalibench.core.protocols import FloatArray, IntArray

Metric = Callable[[FloatArray, IntArray], float]


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "low": self.low, "high": self.high}


def _ci(samples: list[float], value: float, level: float) -> Interval:
    arr = np.asarray([s for s in samples if np.isfinite(s)])
    if len(arr) == 0:
        return Interval(value, float("nan"), float("nan"))
    alpha = (1 - level) / 2
    return Interval(value, float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha)))


def paired_bootstrap(
    metrics: Mapping[str, Metric],
    P_raw: FloatArray,
    P_cal: FloatArray,
    y: IntArray,
    iters: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, dict[str, Interval]]:
    """{metric: {"raw": CI, "calibrated": CI, "difference": CI of calibrated - raw}}."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = [rng.integers(0, n, n) for _ in range(iters)]
    out: dict[str, dict[str, Interval]] = {}
    for name, fn in metrics.items():
        raw_v, cal_v = fn(P_raw, y), fn(P_cal, y)
        raw_s: list[float] = []
        cal_s: list[float] = []
        for idx in draws:
            raw_s.append(fn(P_raw[idx], y[idx]))
            cal_s.append(fn(P_cal[idx], y[idx]))
        diff = [c - r for c, r in zip(cal_s, raw_s, strict=True)]
        out[name] = {
            "raw": _ci(raw_s, raw_v, level),
            "calibrated": _ci(cal_s, cal_v, level),
            "difference": _ci(diff, cal_v - raw_v, level),
        }
    return out
