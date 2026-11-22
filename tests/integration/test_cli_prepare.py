import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from recalibench.cli import app
from tests.conftest import TOY_CONF

TOY = ["--config-dir", str(TOY_CONF)]


def _run(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, result.output


def test_toy_task_builds_from_config_alone(tmp_path: Path) -> None:
    root = f"paths.root={tmp_path}"
    tasks = "experiment.tasks=[toy_binary]"
    code, out = _run(*TOY, "download", root, tasks)
    assert code == 0, out
    code, out = _run(*TOY, "prepare", root, tasks)
    assert code == 0, out
    assert "toy_binary: 200 rows (fit=120 val=40 test=40)" in out

    frame = pd.read_parquet(tmp_path / "data/processed/toy_binary.parquet")
    assert list(frame.columns) == ["id", "state_json", "label", "split"]
    assert frame["id"].is_unique
    rates = frame.groupby("split")["label"].mean()
    assert rates.max() - rates.min() <= 0.02
    meta = json.loads((tmp_path / "data/processed/toy_binary.meta.json").read_text())
    assert meta["counts"] == {"fit": 120, "val": 40, "test": 40}

    code, out = _run(*TOY, "prepare", root, tasks)
    assert code == 0 and "up to date, skipped" in out
    code, out = _run(*TOY, "prepare", root, tasks, "seed=1")
    assert code == 0 and "200 rows" in out  # a new seed changes the fingerprint


def test_prepare_without_raw_data_says_download_first(tmp_path: Path) -> None:
    code, out = _run(*TOY, "prepare", f"paths.root={tmp_path}", "experiment.tasks=[toy_binary]")
    assert code == 1
    assert "run `recal download` first" in out
