"""App data layer on a toy mock pipeline: test rows, fingerprint guard, policy maths."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from recalibench.app import data
from recalibench.core.config import load_config
from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.pipeline import calibrate, download, evaluate, prepare, score
from recalibench.report.collect import Pair
from tests.conftest import TOY_CONF


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Paths:
    root = tmp_path_factory.mktemp("toy")
    ov = [
        f"paths.root={root}",
        "experiment.tasks=[toy_binary,toy_choice]",
        "experiment.models=[mock]",
        "bootstrap_iters=50",
    ]
    cfg = load_config(ov, extra_conf_dirs=[TOY_CONF])
    reg = Registry(extra_conf_dirs=[TOY_CONF], overrides=ov)
    paths = Paths.from_config(cfg)
    download.download(reg, paths, ["toy"])
    prepare.prepare(reg, paths, cfg)
    score.score(reg, paths, cfg)
    calibrate.calibrate(reg, paths, cfg)
    evaluate.evaluate(reg, paths, cfg)
    return paths


def _pair(paths: Paths, task: str) -> Pair:
    return next(p for p in data.load(paths).pairs if p.task == task)


def test_modes_found_on_disk(toy: Paths) -> None:
    assert data.available_modes(toy.root) == ["mock"]
    assert data.default_mode(toy.root) == "mock"


def test_default_mode_prefers_live(tmp_path: Path) -> None:
    for d in ("artifacts", "artifacts_mock"):
        (tmp_path / d / "t" / "m@1").mkdir(parents=True)
        (tmp_path / d / "t" / "m@1" / "metrics.json").write_text("{}")
    assert data.available_modes(tmp_path) == ["live", "mock"]
    assert data.default_mode(tmp_path) == "live"


def test_no_artifacts_means_no_modes(tmp_path: Path) -> None:
    assert data.available_modes(tmp_path) == []


@pytest.mark.parametrize("task", ["toy_binary", "toy_choice"])
def test_test_rows(toy: Paths, task: str) -> None:
    pair = _pair(toy, task)
    rows = data.test_rows(toy, pair)
    assert len(rows) == pair.n_test and rows["id"].is_unique
    keys = list(pair.metrics["keys"])
    for side in ("raw", "cal"):
        probs = rows[[f"{side}_p_{k}" for k in keys]].to_numpy()
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (rows[f"{side}_choice"] == [keys[i] for i in probs.argmax(axis=1)]).all()
        assert np.allclose(rows[f"{side}_conf"], probs.max(axis=1))
        assert (rows[f"{side}_correct"] == (rows[f"{side}_choice"] == rows["label"])).all()
    assert set(rows["label"]) <= set(keys)
    assert rows["text"].str.len().gt(0).all()  # case text joined from data/processed
    # accuracy agrees with the evaluate stage
    assert rows["cal_correct"].mean() == pytest.approx(
        pair.metrics["calibrated"]["accuracy"]["value"]
    )


def test_missing_processed_data_gives_empty_text(toy: Paths, tmp_path: Path) -> None:
    pair = _pair(toy, "toy_binary")
    moved = tmp_path / "processed.parquet"
    toy.processed("toy_binary").rename(moved)
    try:
        rows = data.test_rows(toy, pair)
    finally:
        moved.rename(toy.processed("toy_binary"))
    assert (rows["text"] == "").all()


def test_stale_calibration_is_refused(toy: Paths) -> None:
    pair = _pair(toy, "toy_binary")
    sel_path = toy.artifact_dir("toy_binary", "mock", "mock-1") / "selected.json"
    original = sel_path.read_text()
    sel: dict[str, Any] = json.loads(original)
    sel["scores_fingerprint"] = "rescored-since"
    sel_path.write_text(json.dumps(sel))
    try:
        with pytest.raises(data.AppDataError, match="recal calibrate"):
            data.test_rows(toy, pair)
    finally:
        sel_path.write_text(original)


def test_policy_point() -> None:
    conf = np.array([0.99, 0.97, 0.96, 0.90, 0.60])
    correct = np.array([True, True, False, True, False])
    p = data.policy(conf, correct, 0.95)
    assert (p.n, p.auto_count, p.review_count) == (5, 3, 2)
    assert p.auto_share == pytest.approx(0.6)
    assert p.auto_error_rate == pytest.approx(1 / 3)


def test_policy_nothing_auto_handled() -> None:
    p = data.policy(np.array([0.5, 0.6]), np.array([True, False]), 0.99)
    assert (p.auto_count, p.auto_share, p.auto_error_rate) == (0, 0.0, None)


def test_policy_curve_is_monotone() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 200)
    correct = rng.uniform(size=200) < conf
    curve = data.policy_curve(conf, correct, [0.5, 0.7, 0.9, 0.99])
    shares = [p.auto_share for p in curve]
    assert shares == sorted(shares, reverse=True) and shares[0] == 1.0


def test_calibrated_rows_missing_is_refused(toy: Paths) -> None:
    pair = _pair(toy, "toy_choice")
    path = toy.artifact_dir("toy_choice", "mock", "mock-1") / "calibrated.parquet"
    original = pd.read_parquet(path)
    test_ids = original.loc[original["split"] == "test", "id"]
    kept = cast(pd.DataFrame, original[original["id"] != test_ids.iloc[0]])
    kept.to_parquet(path)
    try:
        with pytest.raises(data.AppDataError, match="missing test rows"):
            data.test_rows(toy, pair)
    finally:
        original.to_parquet(path)
