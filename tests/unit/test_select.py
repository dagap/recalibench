import numpy as np
import pytest

from recalibench.calibrators.binary import Identity, Isotonic, Platt, Temperature
from recalibench.calibrators.select import Candidate, choose, fit_candidates
from recalibench.core.protocols import FloatArray, IntArray
from tests.helpers import synthetic


def _cand(name: str, losses: list[float], ece: float, parametric: bool = True) -> Candidate:
    c = Candidate(name, Identity(name=name), parametric=parametric)
    c.val_row_losses = np.array(losses)
    c.val_log_loss, c.val_ece = float(np.mean(losses)), ece
    return c


def test_clear_winner_by_log_loss() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.4, 500)
    a, b = _cand("a", list(base), 0.01), _cand("b", list(base + 0.05), 0.0)
    assert choose([a, b]).name == "a"
    assert not b.tied_with_best


def test_parametric_beats_non_parametric_inside_the_noise() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.4, 500)
    noise = rng.normal(0, 0.05, 500)
    noise -= noise.mean()  # isotonic is better by exactly 0.001, well inside one SE (~0.0022)
    iso = _cand("isotonic", list(base - 0.001 + noise), 0.001, parametric=False)
    temp = _cand("temperature", list(base), 0.02)
    assert choose([iso, temp]).name == "temperature"
    assert temp.tied_with_best and iso.tied_with_best


def test_ece_breaks_ties_among_parametric() -> None:
    same = [0.3] * 100
    assert choose([_cand("a", same, 0.03), _cand("b", same, 0.01)]).name == "b"


def test_failed_candidates_are_ignored_and_all_failed_raises() -> None:
    bad = _cand("bad", [0.0] * 10, 0.0)
    bad.error = "boom"
    good = _cand("good", [0.5] * 10, 0.1)
    assert choose([bad, good]).name == "good"
    with pytest.raises(ValueError, match="no calibrator"):
        choose([bad])


class Broken:
    supports = frozenset({"binary"})

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, P: FloatArray, y: IntArray) -> None:
        return None

    def predict(self, P: FloatArray) -> FloatArray:
        return P * 2  # rows no longer sum to 1


def test_fit_candidates_on_real_distortion_and_broken_plugin() -> None:
    P_fit, y_fit = synthetic.binary(3000, slope=0.6, seed=2)
    P_val, y_val = synthetic.binary(1000, slope=0.6, seed=3)
    cals = {
        "identity": Identity(name="identity"),
        "temperature": Temperature(name="temperature"),
        "platt": Platt(name="platt"),
        "isotonic": Isotonic(name="isotonic"),
        "broken": Broken(name="broken"),
    }
    cands = fit_candidates(cals, P_fit, y_fit, P_val, y_val, bins=15)
    by = {c.name: c for c in cands}
    assert by["broken"].error is not None and "probability distributions" in by["broken"].error
    assert by["temperature"].val_log_loss < by["identity"].val_log_loss
    assert choose(cands).name in {"temperature", "platt"}  # the family that matches the distortion
    assert by["temperature"].params["T"] == pytest.approx(0.6, abs=0.06)


def test_identity_wins_on_calibrated_scores() -> None:
    P_fit, y_fit = synthetic.binary(3000, seed=4)
    P_val, y_val = synthetic.binary(600, seed=5)
    cals = {
        "identity": Identity(name="identity"),
        "platt": Platt(name="platt"),
        "isotonic": Isotonic(name="isotonic"),
    }
    winner = choose(fit_candidates(cals, P_fit, y_fit, P_val, y_val, bins=15))
    assert winner.name in {"identity", "platt"} and winner.parametric
