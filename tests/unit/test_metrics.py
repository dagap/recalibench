from typing import Any, cast

import numpy as np
import pytest
from sklearn.calibration import calibration_curve  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from sklearn.metrics import (  # pyright: ignore[reportMissingTypeStubs]
    brier_score_loss,  # pyright: ignore[reportUnknownVariableType]
    roc_auc_score,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.metrics import log_loss as sk_log_loss  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from recalibench.evaluate.metrics import (
    accuracy,
    auroc,
    binned_ece,
    brier,
    classwise_ece,
    log_loss,
    reliability,
    smooth_ece,
    top_label_ece,
)
from tests.helpers import synthetic
from tests.helpers.calibration import ece as reference_ece


def test_log_loss_and_accuracy() -> None:
    P = np.array([[0.8, 0.2], [0.4, 0.6]])
    y = np.array([0, 0])
    assert log_loss(P, y) == pytest.approx(-(np.log(0.8) + np.log(0.4)) / 2)
    assert accuracy(P, y) == 0.5


def test_brier_binary_matches_the_one_column_closed_form() -> None:
    P, y = synthetic.binary(1000, seed=0)
    t = (y == 0).astype(float)
    assert brier(P, y) == pytest.approx(float(((P[:, 0] - t) ** 2).mean()))


def test_brier_multiclass() -> None:
    P = np.array([[0.7, 0.2, 0.1]])
    assert brier(P, np.array([0])) == pytest.approx(0.3**2 + 0.2**2 + 0.1**2)


def test_top_label_ece_matches_the_phase4_reference() -> None:
    P, y = synthetic.multiclass(3000, temperature=0.5, seed=1)
    conf, hit = P.max(axis=1), (P.argmax(axis=1) == y).astype(float)
    assert top_label_ece(P, y, 15) == pytest.approx(reference_ece(conf, hit, 15))


def test_calibrated_data_has_small_ece() -> None:
    P, y = synthetic.multiclass(20000, seed=2)
    assert top_label_ece(P, y) < 0.015


@pytest.mark.parametrize("scheme", ["width", "mass"])
def test_perfectly_calibrated_ece_is_near_zero_and_shrinks_with_n(scheme: str) -> None:
    big = synthetic.multiclass(50000, seed=3)
    small = synthetic.multiclass(1000, seed=3)
    assert top_label_ece(*big, scheme=scheme) < 0.01
    assert top_label_ece(*small, scheme=scheme) > top_label_ece(*big, scheme=scheme)
    assert classwise_ece(*big) < 0.01


def test_brier_closed_form_for_a_constant_prediction() -> None:
    rng = np.random.default_rng(4)
    q, p = 0.3, 0.6
    t = rng.random(200000) < q
    P = np.column_stack([np.full(len(t), p), np.full(len(t), 1 - p)])
    y = np.where(t, 0, 1)
    assert brier(P, y) == pytest.approx((p - q) ** 2 + q * (1 - q), abs=2e-3)


def test_metrics_agree_with_scikit_learn() -> None:
    P, y = synthetic.binary(3000, slope=0.7, seed=5)
    t = (y == 0).astype(int)
    assert brier(P, y) == pytest.approx(float(cast(Any, brier_score_loss(t, P[:, 0]))))
    assert log_loss(P, y) == pytest.approx(float(cast(Any, sk_log_loss(t, P[:, 0], labels=[0, 1]))))
    assert auroc(P, y) == pytest.approx(float(cast(Any, roc_auc_score(t, P[:, 0]))))
    # Reliability bins: same observed rate and mean confidence per non-empty bin.
    prob_true, prob_pred = cast(Any, calibration_curve(t, P[:, 0], n_bins=10, strategy="uniform"))
    bins = reliability(P[:, 0], t.astype(float), bins=10)
    assert [b["observed"] for b in bins] == pytest.approx(np.asarray(prob_true).tolist())
    assert [b["confidence"] for b in bins] == pytest.approx(np.asarray(prob_pred).tolist())


def test_auroc_handles_ties_and_one_class() -> None:
    P = np.array([[0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
    y = np.array([0, 1, 0, 1])
    t = (y == 0).astype(int)
    assert auroc(P, y) == pytest.approx(float(cast(Any, roc_auc_score(t, P[:, 0]))))
    assert np.isnan(auroc(P, np.zeros(4, dtype=np.int64)))


def test_binary_classwise_ece_is_the_p_yes_ece() -> None:
    P, y = synthetic.binary(5000, slope=0.6, seed=6)
    assert classwise_ece(P, y) == pytest.approx(binned_ece(P[:, 0], (y == 0).astype(float)))


def test_smooth_ece_small_when_calibrated_large_when_not() -> None:
    good = synthetic.binary(5000, seed=7)
    bad = synthetic.binary(5000, slope=2.5, seed=7)
    assert smooth_ece(*good) < 0.02 < smooth_ece(*bad)
    multi_good, multi_bad = (
        synthetic.multiclass(5000, seed=7),
        synthetic.multiclass(5000, temperature=0.5, seed=7),
    )
    assert smooth_ece(*multi_good) < smooth_ece(*multi_bad)
