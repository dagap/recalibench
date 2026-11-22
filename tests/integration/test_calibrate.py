"""Calibrate stage on the toy tasks: fit on fit, choose on val, never learn from test."""

import json
import pickle
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from recalibench.core.config import load_config
from recalibench.core.paths import Paths
from recalibench.core.protocols import FloatArray, IntArray
from recalibench.core.registry import Registry
from recalibench.pipeline import calibrate, download, prepare, score
from recalibench.pipeline.scores_io import read_scores
from tests.conftest import TOY_CONF


class SpyCal:
    """Identity that records the matrices it is fitted on."""

    supports = frozenset({"binary", "multiclass"})
    fitted: ClassVar[list[FloatArray]] = []

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, P: FloatArray, y: IntArray) -> None:
        SpyCal.fitted.append(P.copy())

    def predict(self, P: FloatArray) -> FloatArray:
        return P


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Registry, Paths, Any]:
    conf = tmp_path / "conf" / "calibrator"
    conf.mkdir(parents=True)
    (conf / "spy.yaml").write_text("_target_: tests.integration.test_calibrate.SpyCal\n")
    dirs = [TOY_CONF, tmp_path / "conf"]
    ov = [
        f"paths.root={tmp_path}",
        "experiment.tasks=[toy_binary,toy_choice]",
        "experiment.models=[tfidf_logreg]",
    ]
    cfg = load_config(ov, extra_conf_dirs=dirs)
    reg = Registry(extra_conf_dirs=dirs, overrides=ov)
    paths = Paths.from_config(cfg)
    download.download(reg, paths, ["toy"])
    prepare.prepare(reg, paths, cfg)
    score.score(reg, paths, cfg)
    return reg, paths, cfg


def _dir(paths: Paths, task: str) -> Path:
    return paths.artifact_dir(task, "tfidf_logreg", "tfidf-lr-1")


def test_outputs_and_candidates(setup: tuple[Registry, Paths, Any]) -> None:
    reg, paths, cfg = setup
    results = {r.task: r for r in calibrate.calibrate(reg, paths, cfg)}
    assert {r.status for r in results.values()} == {"calibrated"}

    binary = json.loads((_dir(paths, "toy_binary") / "selected.json").read_text())
    assert set(binary["candidates"]) == {
        "identity",
        "temperature",
        "platt",
        "beta",
        "isotonic",
        "histogram",
        "toy_identity",
        "spy",
    }
    choice = json.loads((_dir(paths, "toy_choice") / "selected.json").read_text())
    assert set(choice["candidates"]) == {
        "identity",
        "temperature",
        "vector_scaling",
        "dirichlet",
        "toy_identity",
        "spy",
    }
    for sel, task in [(binary, "toy_binary"), (choice, "toy_choice")]:
        d = _dir(paths, task)
        assert sel["selected"] in sel["candidates"] and sel["rule"]
        assert sel["scores_fingerprint"] == read_scores(d).meta["fingerprint"]
        assert sel["n_fit"] == 120 and sel["n_val"] == 40
        cal = pickle.loads((d / "calibrators" / f"{sel['selected']}.pkl").read_bytes())
        P = read_scores(d).probs()
        assert np.allclose(cal.predict(P).sum(axis=1), 1.0)
        out = pd.read_parquet(d / "calibrated.parquet")
        assert set(out["calibrator"]) == set(sel["candidates"])
        assert len(out) == 200 * len(sel["candidates"])  # every calibrator applied to every split


def test_calibrators_are_fitted_on_the_fit_split_only(setup: tuple[Registry, Paths, Any]) -> None:
    reg, paths, cfg = setup
    SpyCal.fitted.clear()
    calibrate.calibrate(reg, paths, cfg)
    for task, fitted in zip(["toy_binary", "toy_choice"], SpyCal.fitted, strict=True):
        assert np.array_equal(fitted, read_scores(_dir(paths, task)).probs("fit"))


def test_poisoning_the_test_split_changes_nothing(setup: tuple[Registry, Paths, Any]) -> None:
    reg, paths, cfg = setup
    calibrate.calibrate(reg, paths, cfg)
    d = _dir(paths, "toy_binary")
    before = json.loads((d / "selected.json").read_text())

    frame = pd.read_parquet(d / "scores.parquet")
    test = frame["split"] == "test"
    frame.loc[test, "label"] = 1 - frame.loc[test, "label"]  # every test label flipped
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, int(test.sum()))
    frame.loc[test, "p_yes"], frame.loc[test, "p_no"] = p, 1 - p
    frame.to_parquet(d / "scores.parquet", index=False)

    calibrate.calibrate(reg, paths, cfg, force=True)
    after = json.loads((d / "selected.json").read_text())
    assert after["selected"] == before["selected"]
    for name, cand in before["candidates"].items():
        assert after["candidates"][name]["val_log_loss"] == pytest.approx(cand["val_log_loss"])


def test_second_run_is_skipped_and_changed_scores_refit(setup: tuple[Registry, Paths, Any]) -> None:
    reg, paths, cfg = setup
    calibrate.calibrate(reg, paths, cfg)
    assert {r.status for r in calibrate.calibrate(reg, paths, cfg)} == {"fresh"}
    # New scores (e.g. a re-score after a model change) carry a new fingerprint:
    d = _dir(paths, "toy_binary")
    meta = json.loads((d / "meta.json").read_text())
    meta["fingerprint"] = "changed"
    (d / "meta.json").write_text(json.dumps(meta))
    statuses = {r.task: r.status for r in calibrate.calibrate(reg, paths, cfg)}
    assert statuses == {"toy_binary": "calibrated", "toy_choice": "fresh"}
