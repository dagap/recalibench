"""Phase 6 checks on synthetic data with a known miscalibration.

Each calibrator is fitted on 5,000 rows and judged on 20,000 fresh rows, and is
only held to distortions its family can represent (temperature can't undo a bias).
"""

from typing import Any

import numpy as np
import pytest

from recalibench.calibrators.binary import Beta, Histogram, Identity, Isotonic, Platt, Temperature
from recalibench.calibrators.multiclass import Dirichlet, VectorScaling
from recalibench.core.protocols import Calibrator
from recalibench.evaluate.metrics import log_loss, top_label_ece
from tests.helpers import synthetic

N_FIT, N_TEST = 5000, 20000
BIAS5 = np.array([1.0, 0.5, 0.0, -0.5, -1.0])

BINARY_CASES: dict[str, dict[str, Any]] = {
    "under-confident": {"slope": 0.6},
    "over-confident": {"slope": 2.0},
    "biased": {"slope": 1.0, "bias": 1.0},
}
BINARY_COVERS = {
    Temperature: ["under-confident", "over-confident"],
    Platt: list(BINARY_CASES),
    Beta: list(BINARY_CASES),
}
MULTI_CASES: dict[str, dict[str, Any]] = {
    "over-confident": {"temperature": 0.5},
    "class-biased": {"class_bias": BIAS5},
}
MULTI_COVERS = {
    Temperature: ["over-confident"],
    VectorScaling: list(MULTI_CASES),
    Dirichlet: list(MULTI_CASES),
}


def _cut(
    cal: Calibrator, fit: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    cal.fit(*fit)
    raw = top_label_ece(*test)
    after = top_label_ece(cal.predict(test[0]), test[1])
    return raw, after


@pytest.mark.parametrize(("cls", "case"), [(c, k) for c, ks in BINARY_COVERS.items() for k in ks])
def test_parametric_binary_calibrators_halve_ece(cls: type, case: str) -> None:
    kw = BINARY_CASES[case]
    raw, after = _cut(
        cls(name="c"), synthetic.binary(N_FIT, **kw, seed=1), synthetic.binary(N_TEST, **kw, seed=2)
    )
    assert raw > 0.03, "the distortion should be visible"
    assert after <= 0.5 * raw, f"{cls.__name__} on {case}: ECE {raw:.4f} -> {after:.4f}"


@pytest.mark.parametrize(("cls", "case"), [(c, k) for c, ks in MULTI_COVERS.items() for k in ks])
def test_parametric_multiclass_calibrators_halve_ece(cls: type, case: str) -> None:
    kw = MULTI_CASES[case]
    raw, after = _cut(
        cls(name="c"),
        synthetic.multiclass(N_FIT, **kw, seed=1),
        synthetic.multiclass(N_TEST, **kw, seed=2),
    )
    assert raw > 0.03
    assert after <= 0.5 * raw, f"{cls.__name__} on {case}: ECE {raw:.4f} -> {after:.4f}"


@pytest.mark.parametrize("cls", [Isotonic, Histogram])
def test_non_parametric_binary_calibrators_also_help(cls: type) -> None:
    kw = BINARY_CASES["under-confident"]
    raw, after = _cut(
        cls(name="c"), synthetic.binary(N_FIT, **kw, seed=1), synthetic.binary(N_TEST, **kw, seed=2)
    )
    assert after < raw


def test_isotonic_output_is_monotone_and_bounded() -> None:
    iso = Isotonic(name="iso")
    iso.fit(*synthetic.binary(N_FIT, slope=0.6, seed=3))
    grid = np.linspace(0, 1, 2001)
    q = iso.predict(np.column_stack([grid, 1 - grid]))[:, 0]
    assert np.all(np.diff(q) >= -1e-12)
    assert q.min() >= 1e-6 and q.max() <= 1 - 1e-6


@pytest.mark.parametrize(
    "data",
    [synthetic.binary(N_TEST, seed=4), synthetic.multiclass(N_TEST, seed=4)],
    ids=["binary", "multiclass"],
)
def test_temperature_on_calibrated_data_stays_near_one(data: tuple[np.ndarray, np.ndarray]) -> None:
    t = Temperature(name="t")
    t.fit(*data)
    assert abs(t.T - 1.0) < 0.05
    assert log_loss(t.predict(data[0]), data[1]) <= log_loss(*data) + 1e-6


def test_temperature_recovers_the_known_distortion() -> None:
    t = Temperature(name="t")
    t.fit(*synthetic.binary(N_TEST, slope=0.6, seed=5))
    assert pytest.approx(0.6, abs=0.03) == t.T  # undoing slope 0.6 means dividing by 0.6


def test_beta_stays_monotone() -> None:
    b = Beta(name="b")
    b.fit(*synthetic.binary(N_FIT, slope=2.0, seed=6))
    assert b.a >= 0 and b.b >= 0
    grid = np.linspace(0.001, 0.999, 999)
    q = b.predict(np.column_stack([grid, 1 - grid]))[:, 0]
    assert np.all(np.diff(q) >= -1e-12)


@pytest.mark.parametrize(
    "cls", [Identity, Temperature, Platt, Beta, Isotonic, Histogram, VectorScaling, Dirichlet]
)
def test_outputs_are_valid_probability_rows(cls: type[Any]) -> None:
    binary_only = "multiclass" not in cls.supports
    P, y = synthetic.binary(500, seed=7) if binary_only else synthetic.multiclass(500, seed=7)
    cal: Calibrator = cls(name="c")
    assert isinstance(cal, Calibrator)
    cal.fit(P, y)
    Q = cal.predict(P)
    assert Q.shape == P.shape
    assert np.allclose(Q.sum(axis=1), 1.0) and Q.min() >= 1e-6 and Q.max() <= 1 - 1e-6


def test_binary_calibrators_reject_multiclass_input() -> None:
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        Platt(name="p").fit(*synthetic.multiclass(50, seed=8))


def test_dirichlet_picks_regularisation_by_cv() -> None:
    d = Dirichlet(name="d")
    d.fit(*synthetic.multiclass(2000, class_bias=BIAS5, seed=9))
    assert d.lam in d.lambdas and len(d.params()["W"]) == 5
