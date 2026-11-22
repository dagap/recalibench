"""Experiments on the toy plug-ins with the real mock model: offline, seconds."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from recalibench.cli import app
from recalibench.models.mock import MockModel
from recalibench.pipeline import experiments as experiments_stage
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


def _summaries(root: Path) -> dict[str, object]:
    exp = root / "artifacts_mock" / "_experiments"
    out: dict[str, object] = {}
    for name in ("wording", "option_order", "transfer"):
        s = json.loads((exp / name / "summary.json").read_text())
        out[name] = s
    return out


def test_batched_variants_match_unbatched_with_fewer_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0, "experiments": 0}
    fake = MockModel.fake_response
    init = MockModel.__init__
    run_experiments = experiments_stage.run_experiments

    def counting(self: MockModel, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        return fake(self, *args, **kwargs)

    def counted_stage(*args: Any, **kwargs: Any) -> Any:
        before = calls["n"]
        out = run_experiments(*args, **kwargs)
        calls["experiments"] = calls["n"] - before
        return out

    def batching_init(self: MockModel, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        self.batch_questions = True

    monkeypatch.setattr(MockModel, "fake_response", counting)
    monkeypatch.setattr(experiments_stage, "run_experiments", counted_stage)
    plain, batched = tmp_path / "plain", tmp_path / "batched"
    _cli(plain, "run")
    unbatched_calls = calls["experiments"]
    monkeypatch.setattr(MockModel, "__init__", batching_init)
    _cli(batched, "run")
    batched_calls = calls["experiments"]

    assert _summaries(plain) == _summaries(batched)
    # toy: 200 items x 3 repeats (mock sends repeats separately). Variant 0 of each
    # experiment is the task's own question, already cached by the main matrix, so
    # wording has 2 new phrasings and option_order 1 new order.
    assert unbatched_calls == 200 * 3 * (2 + 1)
    assert batched_calls == 200 * 3 * 2  # one request per item, repeat and experiment
