import numpy as np
import pytest

from recalibench.evaluate.metrics import accuracy, brier, log_loss, top_label_ece
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
