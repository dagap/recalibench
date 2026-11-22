from pathlib import Path

from typer.testing import CliRunner

from recalibench.cli import app
from tests.conftest import TOY_CONF


def test_list_shows_toys_from_extra_config_dir() -> None:
    result = CliRunner().invoke(app, ["--config-dir", str(TOY_CONF), "list"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "tests.plugins.toy_dataset.ToyDataset" in out
    assert "toy_binary" in out and "dataset=toy kind=binary" in out
    assert "toy@0.1 supports=['binary'] trainable=False" in out
    assert "toy_identity" in out and "supports=['binary', 'multiclass']" in out


def test_list_without_extra_dir_is_empty_but_ok() -> None:
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    assert result.output.count("(none)") == 4


def test_list_fails_on_broken_plugin(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "broken.yaml").write_text("_target_: no.such.Model\n")
    result = CliRunner().invoke(app, ["--config-dir", str(tmp_path), "list"])
    assert result.exit_code == 1
    assert "broken" in result.output and "ERROR" in result.output
