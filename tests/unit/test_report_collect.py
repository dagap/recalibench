from pathlib import Path

import pytest

from recalibench.core.paths import Paths
from recalibench.report.collect import ReportError, ci, collect
from tests.unit.report_fixtures import wording_summary, write_metrics, write_summary


def _arts(tmp_path: Path) -> Path:
    return tmp_path / "artifacts_mock"


def test_collect_finds_every_evaluated_pair(tmp_path: Path) -> None:
    for task in ("t_b", "t_a"):
        for model in ("m2@1", "m1@1"):
            write_metrics(_arts(tmp_path), task=task, model_id=model)
    data = collect(Paths(tmp_path, "mock"))
    assert [(p.task, p.model_id) for p in data.pairs] == [
        ("t_a", "m1@1"),
        ("t_a", "m2@1"),
        ("t_b", "m1@1"),
        ("t_b", "m2@1"),
    ]
    assert data.tasks == ["t_a", "t_b"]
    assert data.models == ["m1@1", "m2@1"]
    assert data.mode == "mock"
    p = data.pairs[0]
    assert (p.kind, p.dataset, p.selected, p.n_test) == ("binary", "toy", "platt", 600)


def test_pair_without_metrics_is_listed_not_evaluated(tmp_path: Path) -> None:
    write_metrics(_arts(tmp_path), task="t", model_id="m@1")
    bare = _arts(tmp_path) / "t" / "n@2"
    bare.mkdir(parents=True)
    (bare / "scores.parquet").write_bytes(b"")
    data = collect(Paths(tmp_path, "mock"))
    assert [p.model_id for p in data.pairs] == ["m@1"]
    assert data.not_evaluated == [("t", "n@2", "no metrics.json")]


def test_experiments_dir_is_not_a_task(tmp_path: Path) -> None:
    write_metrics(_arts(tmp_path), task="t", model_id="m@1")
    write_summary(_arts(tmp_path), "wording", wording_summary())
    data = collect(Paths(tmp_path, "mock"))
    assert "_experiments" not in data.tasks
    assert data.experiments == {"wording": wording_summary()}


def test_live_mode_reads_live_artifacts_only(tmp_path: Path) -> None:
    write_metrics(_arts(tmp_path), task="t", model_id="m@1")
    with pytest.raises(ReportError):
        collect(Paths(tmp_path, "live"))


def test_empty_artifacts_raise(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="recal evaluate"):
        collect(Paths(tmp_path, "mock"))


def test_ci_shapes() -> None:
    assert ci({"value": 0.1, "low": 0.05, "high": 0.2}) == (0.1, 0.05, 0.2)
    assert ci({"value": 0.3}) == (0.3, None, None)
    assert ci(None) is None


def test_malformed_metrics_listed_not_evaluated(tmp_path: Path) -> None:
    arts = _arts(tmp_path)
    write_metrics(arts, task="t", model_id="good@1")
    for name, text in (("trunc@1", '{"task": "t", "mod'), ("nokeys@1", '{"task": "t"}')):
        d = arts / "t" / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(text)
    data = collect(Paths(tmp_path, "mock"))
    assert [p.model_id for p in data.pairs] == ["good@1"]
    bad = {m: reason for _, m, reason in data.not_evaluated}
    assert set(bad) == {"trunc@1", "nokeys@1"}
    assert all(r.startswith("unreadable metrics.json") for r in bad.values())
