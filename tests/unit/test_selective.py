import numpy as np
import pytest

from recalibench.evaluate.selective import (
    apply_threshold,
    aurc,
    risk_coverage,
    selective_summary,
    threshold_for_accuracy,
)
from tests.helpers import synthetic


def test_risk_coverage_hand_example() -> None:
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1.0, 0.0, 1.0, 1.0])
    cov, risk = risk_coverage(conf, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0.0, 0.5, 1 / 3, 0.25])
    assert aurc(conf, correct) == pytest.approx((0 + 0.5 + 1 / 3 + 0.25) / 4)


def test_perfect_ranking_has_zero_risk_up_to_the_error_free_share() -> None:
    conf = np.linspace(1, 0, 100)
    correct = np.r_[np.ones(70), np.zeros(30)]
    cov, risk = risk_coverage(conf, correct)
    assert np.all(risk[cov <= 0.7] == 0) and risk[-1] == pytest.approx(0.3)


def test_threshold_for_accuracy_and_apply() -> None:
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.55])
    correct = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    assert threshold_for_accuracy(conf, correct, 0.95) == 0.9  # top 3 are all right
    assert threshold_for_accuracy(conf, correct, 1.01) is None
    pol = apply_threshold(conf, correct, 0.9)
    assert (pol.coverage, pol.accuracy) == (0.6, 1.0)
    assert apply_threshold(conf, correct, None).coverage == 0.0


def test_trust_policy_misses_target_when_overconfident_hits_after_fixing() -> None:
    over_val, yv = synthetic.binary(4000, slope=2.5, seed=0)
    over_test, yt = synthetic.binary(20000, slope=2.5, seed=1)
    good_val, _ = synthetic.binary(4000, slope=1.0, seed=0)
    good_test, _ = synthetic.binary(20000, slope=1.0, seed=1)
    raw = selective_summary(over_val, yv, over_test, yt)
    cal = selective_summary(good_val, yv, good_test, yt)
    assert raw["trust"]["accuracy"] < 0.95 - 0.01  # "95% sure" isn't 95% right
    assert cal["trust"]["accuracy"] >= 0.95 - 0.005
    assert abs(raw["tuned"]["accuracy"] - 0.95) < 0.02  # tuning on val rescues raw
    assert len(raw["curve"]["coverage"]) == 50
