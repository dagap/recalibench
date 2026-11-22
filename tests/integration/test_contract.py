"""Plug-in contract: toy plug-ins defined only under tests/plugins/ run end to end
through the real CLI, with no change to src/. Grows with each phase (calibrate,
evaluate and report join later)."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from recalibench.cli import app
from recalibench.pipeline.scores_io import CLIP, read_scores
from tests.conftest import TOY_CONF

SRC = Path(__file__).resolve().parents[2] / "src"
MODELS = "experiment.models=[toy,tfidf_logreg,structured_logreg]"
TASKS = "experiment.tasks=[toy_binary,toy_choice]"


def _cli(tmp_path: Path, *args: str) -> str:
    result = CliRunner().invoke(
        app, ["--config-dir", str(TOY_CONF), *args, f"paths.root={tmp_path}", TASKS, MODELS]
    )
    assert result.exit_code == 0, result.output
    return result.output


def _auroc(p: np.ndarray, y: np.ndarray) -> float:
    ranks = p.argsort().argsort() + 1
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def test_toy_plugins_run_end_to_end_through_the_cli(tmp_path: Path) -> None:
    _cli(tmp_path, "download")
    _cli(tmp_path, "prepare")
    out = _cli(tmp_path, "score")

    arts = tmp_path / "artifacts_mock"
    expected = {
        ("toy_binary", "toy@0.1"),
        ("toy_binary", "tfidf_logreg@tfidf-lr-1"),
        ("toy_binary", "structured_logreg@struct-lr-1"),
        ("toy_choice", "tfidf_logreg@tfidf-lr-1"),
    }
    assert {(d.parent.name, d.name) for d in arts.glob("*/*")} == expected
    assert "skipped   toy_choice x toy@0.1" in out
    assert "skipped   toy_choice x structured_logreg@struct-lr-1" in out

    for task, model in expected:
        s = read_scores(arts / task / model)
        assert len(s.frame) == 200 and s.frame["id"].is_unique
        assert set(s.frame["split"]) == {"fit", "val", "test"}
        p = s.probs()
        assert p.min() >= CLIP and p.max() <= 1 - CLIP
        assert np.allclose(p.sum(axis=1), 1.0)

    s = read_scores(arts / "toy_binary" / "structured_logreg@struct-lr-1")
    assert _auroc(s.probs()[:, 0], s.labels().to_numpy()) > 0.8  # the toy signal is learnable
    assert s.meta["out_of_fold"] is True

    again = _cli(tmp_path, "score")  # a second run re-scores nothing
    assert "  scored " not in again and again.count("  fresh ") == len(expected)


def test_src_never_references_test_plugins() -> None:
    offenders = [p for p in SRC.rglob("*.py") if "tests.plugins" in p.read_text()]
    assert offenders == []
