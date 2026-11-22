from typer.testing import CliRunner

from recalibench.cli import app
from recalibench.core.config import load_config

COMMANDS = [
    "list",
    "download",
    "prepare",
    "score",
    "calibrate",
    "evaluate",
    "experiments",
    "report",
    "run",
]


def test_cli_help_lists_all_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_config_tree_composes() -> None:
    cfg = load_config(["mode=mock"])
    assert cfg.mode == "mock"
    assert len(cfg.experiment.tasks) == 3
