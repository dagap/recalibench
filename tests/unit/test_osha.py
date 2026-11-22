import json
import shutil
from pathlib import Path

import httpx
import pytest
import respx

from recalibench.core.schema import check_schema
from recalibench.datasets.manifest import MANIFEST, DownloadError, sha256_file
from recalibench.datasets.osha import ZIP_NAME, OshaSIR

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "osha_sir"
URL = "https://example.test/sir.zip"
FIXTURE_SHA = sha256_file(FIXTURE / ZIP_NAME)


def _ds(sha: str | None = FIXTURE_SHA) -> OshaSIR:
    return OshaSIR(name="osha_sir", url=URL, sha256=sha, timeout_s=1)


def test_fixture_matches_schema() -> None:
    ds = _ds()
    df = ds.load(FIXTURE)
    assert len(df) == 50
    assert check_schema(df, ds.schema()) == []
    assert df["id"].str.startswith("osha-").all()


def test_fixture_is_anonymised() -> None:
    df = _ds().load(FIXTURE)
    assert df["Employer"].str.fullmatch(r"Employer \d{3}").all()
    for col in ["Address1", "Address2", "City", "Zip", "Latitude", "Longitude"]:
        assert df[col].isna().all(), col


@respx.mock
def test_download_writes_zip_and_manifest(tmp_path: Path) -> None:
    route = respx.get(URL).mock(
        return_value=httpx.Response(200, content=(FIXTURE / ZIP_NAME).read_bytes())
    )
    _ds().download(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    assert manifest["sha256"] == FIXTURE_SHA
    assert manifest["source"] == URL
    assert "research-client" in route.calls.last.request.headers["user-agent"]
    assert not (tmp_path / "sir.part").exists()


@respx.mock
def test_second_download_makes_no_request(tmp_path: Path) -> None:
    route = respx.get(URL).mock(
        return_value=httpx.Response(200, content=(FIXTURE / ZIP_NAME).read_bytes())
    )
    _ds().download(tmp_path)
    _ds().download(tmp_path)
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize(
    "response",
    [httpx.Response(403), httpx.Response(500), httpx.TimeoutException("slow")],
)
def test_failure_explains_manual_placement(
    tmp_path: Path, response: httpx.Response | Exception
) -> None:
    if isinstance(response, Exception):
        respx.get(URL).mock(side_effect=response)
    else:
        respx.get(URL).mock(return_value=response)
    with pytest.raises(DownloadError) as exc:
        _ds().download(tmp_path)
    msg = str(exc.value)
    assert str(tmp_path / ZIP_NAME) in msg
    assert "severe-injury-reports" in msg
    assert FIXTURE_SHA in msg
    assert not (tmp_path / ZIP_NAME).exists()


@respx.mock
def test_checksum_mismatch_rejected(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"not the pinned file"))
    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        _ds().download(tmp_path)
    assert not (tmp_path / ZIP_NAME).exists()


@respx.mock
def test_hand_placed_zip_is_accepted_without_network(tmp_path: Path) -> None:
    route = respx.get(URL)
    shutil.copy(FIXTURE / ZIP_NAME, tmp_path / ZIP_NAME)
    _ds().download(tmp_path)
    assert route.call_count == 0
    assert "placed by hand" in json.loads((tmp_path / MANIFEST).read_text())["source"]
