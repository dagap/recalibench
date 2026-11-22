from collections import Counter

import numpy as np
import pytest

from recalibench.core.splits import largest_remainder, stratified_sample, stratified_split

FRACTIONS = {"fit": 0.6, "val": 0.2, "test": 0.2}


@pytest.mark.parametrize(
    ("total", "weights", "expected"),
    [(10, [1, 1, 1], [4, 3, 3]), (5, [0.6, 0.2, 0.2], [3, 1, 1]), (1, [0.6, 0.2, 0.2], [1, 0, 0])],
)
def test_largest_remainder(total: int, weights: list[float], expected: list[int]) -> None:
    assert largest_remainder(total, weights) == expected


def test_sample_keeps_stratum_shares_exactly() -> None:
    strata = ["a"] * 700 + ["b"] * 300
    idx = stratified_sample(strata, 100, np.random.default_rng(0))
    assert len(idx) == 100 and len(set(idx.tolist())) == 100
    assert Counter(strata[int(i)] for i in idx.tolist()) == {"a": 70, "b": 30}


def test_sample_larger_than_population_takes_all() -> None:
    assert stratified_sample(["a", "b"], 10, np.random.default_rng(0)).tolist() == [0, 1]


def test_split_is_exact_per_stratum() -> None:
    strata = ["a"] * 500 + ["b"] * 50 + ["c"] * 3
    split = stratified_split(strata, FRACTIONS, np.random.default_rng(1))
    for key, n in [("a", 500), ("b", 50)]:
        got = Counter(s for s, k in zip(split, strata, strict=True) if k == key)
        assert got == {"fit": round(n * 0.6), "val": round(n * 0.2), "test": round(n * 0.2)}
    tiny = Counter(s for s, k in zip(split, strata, strict=True) if k == "c")
    assert sum(tiny.values()) == 3 and tiny["fit"] >= 1


def test_split_is_deterministic_per_seed() -> None:
    strata = [str(i % 4) for i in range(400)]
    a = stratified_split(strata, FRACTIONS, np.random.default_rng(42))
    b = stratified_split(strata, FRACTIONS, np.random.default_rng(42))
    c = stratified_split(strata, FRACTIONS, np.random.default_rng(43))
    assert (a == b).all() and not (a == c).all()


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split(["a"], {"fit": 0.5, "val": 0.2}, np.random.default_rng(0))
