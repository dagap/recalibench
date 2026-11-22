"""Fit every candidate calibrator on the fit split and choose one on val.

Rule (recorded in selected.json):
  1. lowest val log loss wins;
  2. a candidate is tied with the best when its paired per-row val loss
     difference is within one standard error of zero (the "one-SE rule": with
     ~600 val rows, smaller gaps are noise), or within TIE_REL; among tied
     candidates a parametric family beats a non-parametric one (a small split
     overfits isotonic or histogram binning; PLAN.md risk table);
  3. remaining ties go to the lowest val ECE, then to registry order.
The test split is never passed in here.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from recalibench.core.protocols import Calibrator, FloatArray, IntArray
from recalibench.evaluate.metrics import brier, log_loss, top_label_ece

TIE_REL = 1e-3
RULE = (
    "lowest val log loss; candidates within one standard error of the best (paired per-row "
    "difference) are tied; among ties, parametric before non-parametric, then lowest val "
    "top-label ECE"
)


@dataclass
class Candidate:
    name: str
    calibrator: Calibrator
    parametric: bool
    val_log_loss: float = float("nan")
    val_ece: float = float("nan")
    val_brier: float = float("nan")
    fit_seconds: float = 0.0
    error: str | None = None
    params: dict[str, Any] = field(default_factory=lambda: {})
    val_row_losses: FloatArray = field(default_factory=lambda: np.array([]))
    tied_with_best: bool = False

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "parametric": self.parametric,
            "val_log_loss": self.val_log_loss,
            "val_ece": self.val_ece,
            "val_brier": self.val_brier,
            "fit_seconds": round(self.fit_seconds, 4),
            "params": self.params,
            "tied_with_best": self.tied_with_best,
        }
        if self.error:
            out["error"] = self.error
        return out


def val_metrics(P: FloatArray, y: IntArray, bins: int) -> dict[str, float]:
    return {"log_loss": log_loss(P, y), "ece": top_label_ece(P, y, bins), "brier": brier(P, y)}


def _check(Q: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    Q = np.asarray(Q, dtype=np.float64)
    if Q.shape != shape:
        raise ValueError(f"predict returned shape {Q.shape}, expected {shape}")
    if not np.isfinite(Q).all() or Q.min() < 0 or not np.allclose(Q.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("predict returned rows that are not probability distributions")
    return Q


def fit_candidates(
    calibrators: Mapping[str, Calibrator],
    P_fit: FloatArray,
    y_fit: IntArray,
    P_val: FloatArray,
    y_val: IntArray,
    bins: int,
) -> list[Candidate]:
    out: list[Candidate] = []
    for name, cal in calibrators.items():
        cand = Candidate(name, cal, parametric=bool(getattr(cal, "parametric", True)))
        try:
            start = time.perf_counter()
            cal.fit(P_fit, y_fit)
            cand.fit_seconds = time.perf_counter() - start
            Q = _check(cal.predict(P_val), P_val.shape)
            m = val_metrics(Q, y_val, bins)
            cand.val_row_losses = -np.log(np.clip(Q[np.arange(len(y_val)), y_val], 1e-15, 1.0))
            cand.val_log_loss, cand.val_ece, cand.val_brier = m["log_loss"], m["ece"], m["brier"]
            params = getattr(cal, "params", None)
            cand.params = cast(dict[str, Any], params()) if callable(params) else {}
        except Exception as e:  # a broken plug-in must not sink the others
            cand.error = f"{type(e).__name__}: {e}"
        out.append(cand)
    return out


def choose(candidates: list[Candidate]) -> Candidate:
    ok = [c for c in candidates if c.error is None and np.isfinite(c.val_log_loss)]
    if not ok:
        raise ValueError("no calibrator fitted successfully")
    best = min(ok, key=lambda c: c.val_log_loss)
    tied: list[Candidate] = []
    for c in ok:
        diff = c.val_row_losses - best.val_row_losses
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        c.tied_with_best = float(diff.mean()) <= max(se, TIE_REL * abs(best.val_log_loss)) + 1e-12
        if c.tied_with_best:
            tied.append(c)
    if any(c.parametric for c in tied):
        tied = [c for c in tied if c.parametric]
    order = {c.name: i for i, c in enumerate(candidates)}
    return min(tied, key=lambda c: (c.val_ece, c.val_log_loss, order[c.name]))
