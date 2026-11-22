import json
from pathlib import Path

import pandas as pd
import pytest

from recalibench.core.schema import check_schema
from recalibench.datasets import workers_comp as wc
from recalibench.datasets.manifest import MANIFEST, DownloadError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "workers_comp"


def test_fixture_matches_schema() -> None:
    ds = wc.WorkersComp(name="workers_comp")
    df = ds.load(FIXTURE)
    assert len(df) == 50
    assert check_schema(df, ds.schema()) == []


def _openml_like_frame() -> pd.DataFrame:
    """What fetch_openml returns: the fixture without our id column."""
    return pd.read_parquet(FIXTURE / wc.PARQUET_NAME).drop(columns="id")


def test_download_adds_ids_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_fetch(openml_id: int, data_home: Path) -> pd.DataFrame:
        calls.append(openml_id)
        return _openml_like_frame()

    monkeypatch.setattr(wc, "_fetch", fake_fetch)
    ds = wc.WorkersComp(name="workers_comp")
    ds.download(tmp_path)
    ds.download(tmp_path)  # current: no second fetch
    assert calls == [42876]
    df = ds.load(tmp_path)
    assert df["id"].tolist()[:2] == ["wc-000000", "wc-000001"]
    assert check_schema(df, ds.schema()) == []
    assert json.loads((tmp_path / MANIFEST).read_text())["rows"] == 50


def test_fetch_failure_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(openml_id: int, data_home: Path) -> pd.DataFrame:
        raise ConnectionError("offline")

    monkeypatch.setattr(wc, "_fetch", boom)
    with pytest.raises(DownloadError, match="OpenML dataset 42876"):
        wc.WorkersComp(name="workers_comp").download(tmp_path)
