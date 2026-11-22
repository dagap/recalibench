"""Evaluate stage on the toy tasks: metrics.json contents, fingerprint guard, freshness."""

import json
from pathlib import Path
from typing import Any

import pytest

from recalibench.core.config import load_config
from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.pipeline import calibrate, download, evaluate, prepare, score
from tests.conftest import TOY_CONF


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Registry, Paths, Any]:
    ov = [
        f"paths.root={tmp_path}",
        "experiment.tasks=[toy_binary,toy_choice]",
        "experiment.models=[mock]",
        "bootstrap_iters=200",
    ]
    cfg = load_config(ov, extra_conf_dirs=[TOY_CONF])
    reg = Registry(extra_conf_dirs=[TOY_CONF], overrides=ov)
    paths = Paths.from_config(cfg)
    download.download(reg, paths, ["toy"])
    prepare.prepare(reg, paths, cfg)
    score.score(reg, paths, cfg)
    calibrate.calibrate(reg, paths, cfg)
    return reg, paths, cfg


def _metrics(paths: Paths, task: str) -> dict[str, Any]:
    return json.loads((paths.artifact_dir(task, "mock", "mock-1") / "metrics.json").read_text())


def test_metrics_json_contents(setup: tuple[Registry, Paths, Any]) -> None:
    reg, paths, cfg = setup
    assert {r.status for r in evaluate.evaluate(reg, paths, cfg)} == {"evaluated"}
    binary, choice = _metrics(paths, "toy_binary"), _metrics(paths, "toy_choice")
    assert binary["n_test"] == 40 and binary["ci"]["iters"] == 200
    for m in (binary, choice):
        for name in ("log_loss", "brier", "accuracy", "ece", "ece_equal_mass", "classwise_ece"):
            for side in ("raw", "calibrated"):
                ci = m[side][name]
                assert ci["low"] <= ci["high"]
            d = m["difference"][name]
            assert d["value"] == pytest.approx(
                m["calibrated"][name]["value"] - m["raw"][name]["value"]
            )
        assert "smooth_ece" in m["raw"] and m["selected"] in m["candidates"]
        assert m["reliability"]["raw"]["top_label"]
        assert {"tuned", "trust", "aurc", "curve"} <= set(m["selective"]["calibrated"])
    assert "auroc" in binary["raw"] and "auroc" not in choice["raw"]
    assert "p_yes" in binary["reliability"]["raw"]


def test_stale_calibrators_are_refused(setup: tuple[Registry, Paths, Any]) -> None:
    reg, paths, cfg = setup
    d = paths.artifact_dir("toy_binary", "mock", "mock-1")
    meta = json.loads((d / "meta.json").read_text())
    meta["fingerprint"] = "rescored-since"
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(evaluate.StaleCalibration, match="recal calibrate"):
        evaluate.evaluate(reg, paths, cfg)


def test_second_run_is_skipped(setup: tuple[Registry, Paths, Any]) -> None:
    reg, paths, cfg = setup
    evaluate.evaluate(reg, paths, cfg)
    assert {r.status for r in evaluate.evaluate(reg, paths, cfg)} == {"fresh"}
