import numpy as np
import pytest

from recalibench.evaluate.bootstrap import Metric, paired_bootstrap
from recalibench.evaluate.metrics import brier, log_loss, top_label_ece
from tests.helpers import synthetic

FNS: dict[str, Metric] = {"log_loss": log_loss, "brier": brier, "ece": top_label_ece}


def test_same_seed_same_intervals_other_seed_differs() -> None:
    P, y = synthetic.binary(600, slope=0.6, seed=0)
    Q, _ = synthetic.binary(600, seed=0)
    a = paired_bootstrap(FNS, P, Q, y, iters=200, seed=3)
    b = paired_bootstrap(FNS, P, Q, y, iters=200, seed=3)
    c = paired_bootstrap(FNS, P, Q, y, iters=200, seed=4)
    assert a == b
    assert a["log_loss"]["raw"] != c["log_loss"]["raw"]


def test_interval_contains_the_point_and_truth() -> None:
    P, y = synthetic.binary(2000, seed=1)
    out = paired_bootstrap({"brier": brier}, P, P, y, iters=300, seed=0)["brier"]
    raw = out["raw"]
    assert raw.low <= raw.value <= raw.high
    expected = float((P[:, 0] * (1 - P[:, 0])).mean())  # E[Brier] for calibrated scores
    assert raw.low <= expected <= raw.high


def test_identical_scores_give_a_zero_difference() -> None:
    P, y = synthetic.binary(500, seed=2)
    diff = paired_bootstrap({"log_loss": log_loss}, P, P, y, iters=100, seed=0)["log_loss"][
        "difference"
    ]
    assert diff.value == 0 and diff.low == 0 and diff.high == 0


def test_a_real_improvement_has_a_negative_difference_interval() -> None:
    P_raw, y = synthetic.binary(1000, slope=0.4, seed=3)
    P_cal, y2 = synthetic.binary(1000, slope=1.0, seed=3)  # same draws, undistorted
    assert np.array_equal(y, y2)
    diff = paired_bootstrap({"log_loss": log_loss}, P_raw, P_cal, y, iters=300, seed=0)["log_loss"]
    assert diff["difference"].high < 0
    assert diff["calibrated"].value == pytest.approx(log_loss(P_cal, y))
