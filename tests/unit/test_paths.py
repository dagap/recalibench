from pathlib import Path

import pytest

from recalibench.core.config import load_config
from recalibench.core.paths import Paths, model_key


def test_mock_and_live_never_share_cache_or_artifacts(tmp_path: Path) -> None:
    mock, live = Paths(tmp_path, "mock"), Paths(tmp_path, "live")
    assert mock.cache_dir("jev", "jev-1.13.0") == tmp_path / "cache_mock" / "jev@jev-1.13.0"
    assert live.cache_dir("jev", "jev-1.13.0") == tmp_path / "cache" / "jev@jev-1.13.0"
    assert mock.artifact_dir("t", "m", "1") == tmp_path / "artifacts_mock" / "t" / "m@1"
    assert live.artifact_dir("t", "m", "1") == tmp_path / "artifacts" / "t" / "m@1"


def test_data_paths_are_mode_independent(tmp_path: Path) -> None:
    for mode in ("mock", "live"):
        p = Paths(tmp_path, mode)
        assert p.raw_dir("osha_sir") == tmp_path / "data" / "raw" / "osha_sir"
        assert p.processed("t") == tmp_path / "data" / "processed" / "t.parquet"
        assert p.reports_dir() == tmp_path / "reports"


@pytest.mark.parametrize("bad", ["", "..", "a/b", "a b", "x@y"])
def test_unsafe_components_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        model_key(bad, "1")


def test_bad_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Paths(tmp_path, "prod")  # type: ignore[arg-type]


def test_from_config_follows_mode_override() -> None:
    p = Paths.from_config(load_config(["mode=live"]))
    assert p.mode == "live"
    assert p.root.is_absolute()


def test_artifacts_root_per_mode(tmp_path: Path) -> None:
    assert Paths(tmp_path, "mock").artifacts_root() == tmp_path / "artifacts_mock"
    assert Paths(tmp_path, "live").artifacts_root() == tmp_path / "artifacts"
    mock = Paths(tmp_path, "mock")
    assert mock.experiments_root() == tmp_path / "artifacts_mock" / "_experiments"
