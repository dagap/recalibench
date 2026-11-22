"""Stratified sampling and splitting with exact per-stratum counts.

Every stratum is shuffled and cut on its own, with largest-remainder rounding,
so each split keeps the population's stratum shares to within one row per
stratum. numpy only: the same seed and input always give the same result.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

IndexArray = npt.NDArray[np.intp]


def largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    """Integers proportional to `weights` that sum exactly to `total`."""
    w = np.asarray(weights, dtype=np.float64)
    if total < 0 or (w < 0).any() or w.sum() <= 0:
        raise ValueError("need total >= 0 and non-negative weights with a positive sum")
    raw = total * w / w.sum()
    out = np.floor(raw).astype(np.int64)
    short = total - int(out.sum())
    if short:
        order = np.argsort(-(raw - out), kind="stable")  # ties go to the earlier entry
        out[order[:short]] += 1
    return [int(x) for x in out]


def _groups(strata: Sequence[str]) -> list[IndexArray]:
    keys = np.asarray(strata, dtype=object)
    uniq = sorted({str(k) for k in keys})
    return [np.flatnonzero(keys == k) for k in uniq]


def stratified_sample(strata: Sequence[str], n: int, rng: np.random.Generator) -> IndexArray:
    """Indices of `n` rows (or all, if fewer), keeping each stratum's share."""
    if n >= len(strata):
        return np.arange(len(strata), dtype=np.intp)
    groups = _groups(strata)
    sizes = largest_remainder(n, [len(g) for g in groups])
    picked = [rng.choice(g, size=k, replace=False) for g, k in zip(groups, sizes, strict=True)]
    return np.sort(np.concatenate(picked)).astype(np.intp)


def stratified_split(
    strata: Sequence[str], fractions: Mapping[str, float], rng: np.random.Generator
) -> npt.NDArray[np.object_]:
    """A split name per row, e.g. fit/val/test, cut separately inside every stratum."""
    names = list(fractions)
    weights = [float(fractions[k]) for k in names]
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {sum(weights)}")
    out = np.empty(len(strata), dtype=object)
    for group in _groups(strata):
        shuffled = rng.permutation(group)
        sizes = largest_remainder(len(group), weights)
        start = 0
        for name, size in zip(names, sizes, strict=True):
            out[shuffled[start : start + size]] = name
            start += size
    return out
