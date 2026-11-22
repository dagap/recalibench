from pathlib import Path

from typer.testing import CliRunner

from recalibench.cli import app
from tests.conftest import TOY_CONF


def test_download_toy_logs_row_count(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["--config-dir", str(TOY_CONF), "download", "--dataset", "toy", f"paths.root={tmp_path}"],
    )
    assert result.exit_code == 0, result.output
    assert "toy: 200 rows x 4 columns, schema OK" in result.output
    assert (tmp_path / "data" / "raw" / "toy" / "toy.csv").exists()


def test_default_selection_follows_experiment_tasks(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--config-dir",
            str(TOY_CONF),
            "download",
            f"paths.root={tmp_path}",
            "experiment.tasks=[toy_binary]",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "toy: 200 rows" in result.output


def test_unknown_task_hints_dataset_flag(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["download", f"paths.root={tmp_path}", "experiment.tasks=[no_such_task]"]
    )
    assert result.exit_code == 1
    assert "--dataset" in result.output
