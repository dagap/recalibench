"""Experiments on the toy plug-ins with the real mock model: offline, seconds."""

import json
from pathlib import Path

from typer.testing import CliRunner

from recalibench.cli import app
from tests.conftest import TOY_CONF


def _cli(tmp_path: Path, *args: str) -> str:
    result = CliRunner().invoke(
        app,
        [
            "--config-dir",
            str(TOY_CONF),
            *args,
            "experiment=toy",
            f"paths.root={tmp_path}",
            "bootstrap_iters=100",
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_all_experiments_run_offline(tmp_path: Path) -> None:
    _cli(tmp_path, "run")
    out = _cli(tmp_path, "experiments")
    for name in ("wording", "option_order", "transfer", "cross_model"):
        assert f"done      {name}" in out
    root = tmp_path / "artifacts_mock" / "_experiments"

    wording = json.loads((root / "wording" / "summary.json").read_text())
    assert [v["variant"] for v in wording["variants"]] == ["phrasing_0", "phrasing_1", "phrasing_2"]
    assert len({v["question"] for v in wording["variants"]}) == 3
    assert wording["raw_ece_spread"] >= 0

    order = json.loads((root / "option_order" / "summary.json").read_text())
    firsts = [v["first_option"] for v in order["variants"]]
    assert len(firsts) == 2 and len(set(firsts)) == 2  # each order puts a different option first
    for v in order["variants"]:  # 40 test rows: structure here; the effect is tested at scale
        assert v["order"][0] == v["first_option"]  # score columns follow the shown order
        for side in ("raw", "calibrated"):
            assert 0 <= v[f"{side}_first_option_mean_prob"] <= 1
    assert order["calibrated_mean_prob_instability"] >= 0

    transfer = json.loads((root / "transfer" / "summary.json").read_text())
    assert {"raw", "transferred", "native"} <= set(transfer) and transfer["to"] == "toy_binary_text"

    cross = json.loads((root / "cross_model" / "summary.json").read_text())
    assert {r["model_id"] for r in cross["tasks"]["toy_binary"]} == {
        "mock@mock-1",
        "tfidf_logreg@tfidf-lr-1",
    }

    again = _cli(tmp_path, "experiments")  # variants are fresh: nothing is re-scored
    assert "done      wording" in again


def test_dry_run_estimates_the_variants(tmp_path: Path) -> None:
    _cli(tmp_path, "download")
    _cli(tmp_path, "prepare")
    out = _cli(tmp_path, "experiments", "dry_run=true")
    assert "estimated estimate: $0.0000 for 5 variants" in out
