"""Report stage on the toy plug-ins with the real mock model, through the CLI."""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from recalibench.cli import app
from recalibench.core.paths import Paths
from recalibench.models.budget import Budget
from recalibench.pipeline import experiments as experiments_stage
from recalibench.report.collect import collect
from tests.conftest import TOY_CONF


def _invoke(tmp_path: Path, *args: str) -> tuple[int, str]:
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
    return result.exit_code, result.output


def test_run_builds_the_full_report(tmp_path: Path) -> None:
    code, out = _invoke(tmp_path, "run")
    assert code == 0, out
    assert "done      wording" in out  # `recal run` now includes the experiments stage
    report = tmp_path / "reports" / "report_mock.html"
    assert f"built     {report}" in out
    html = report.read_text()

    data = collect(Paths(tmp_path, "mock"))
    figures = 3 * len(data.pairs) + len(data.tasks) + 2  # + wording, option_order charts
    svgs = re.findall(r"<svg.*?</svg>", html, flags=re.S)
    assert len(svgs) == figures
    assert all('id="raw"' in s and 'id="calibrated"' in s for s in svgs)
    for p in data.pairs:
        assert f'id="pair-{p.task}-{p.model_id}"' in html
    for name in ("wording", "option_order", "transfer"):
        assert f'id="exp-{name}"' in html


def test_report_on_its_own_matches_run(tmp_path: Path) -> None:
    assert _invoke(tmp_path, "run")[0] == 0
    code, out = _invoke(tmp_path, "report")
    assert code == 0, out
    assert "built     " in out and "task x model pairs" in out


def test_report_errors_when_nothing_evaluated(tmp_path: Path) -> None:
    code, out = _invoke(tmp_path, "report")
    assert code == 1
    assert "recal evaluate" in out
    assert not (tmp_path / "reports").exists()


def test_run_shares_one_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # One Budget per run, shared by all models (score and experiments alike).
    made: list[Budget] = []
    original = Budget.__post_init__

    def counting(self: Budget) -> None:
        made.append(self)
        original(self)

    monkeypatch.setattr(Budget, "__post_init__", counting)
    code, out = _invoke(tmp_path, "run")
    assert code == 0, out
    assert len(made) == 1


def test_run_with_limit_skips_experiments_and_report(tmp_path: Path) -> None:
    code, out = _invoke(tmp_path, "run", "limit=20")  # make smoke's size; tf-idf needs ~20
    assert code == 0, out
    assert "skipped   experiments and report" in out
    assert not (tmp_path / "reports").exists()


def test_late_stage_failure_keeps_earlier_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise experiments_stage.ExperimentError("boom")

    monkeypatch.setattr(experiments_stage, "run_experiments", boom)
    code, out = _invoke(tmp_path, "run")
    assert code == 1
    assert "  evaluated " in out and "  calibrated " in out  # earlier stages still reported
    assert "boom" in out
