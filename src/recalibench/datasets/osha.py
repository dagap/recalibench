"""OSHA Severe Injury Reports (federal OSHA jurisdiction, 2015 onwards).

Source page: https://www.osha.gov/severe-injury-reports. The full data set is a
ZIP holding one UTF-8 CSV. The URL and its SHA-256 are pinned in
conf/dataset/osha_sir.yaml; OSHA publishes refreshed data under a new file name.
"""

from pathlib import Path

import httpx
import pandas as pd

from recalibench.core.schema import apply_schema
from recalibench.datasets.manifest import DownloadError, is_current, sha256_file, write_manifest

SOURCE_PAGE = "https://www.osha.gov/severe-injury-reports"
ZIP_NAME = "sir.zip"

# Columns as published (checked against January2015toNovember2025.zip), and the
# dtypes load() enforces. pandas 3 reads text as "str".
SCHEMA: dict[str, str] = {
    "id": "str",
    "ID": "int64",
    "UPA": "int64",
    "EventDate": "str",
    "Employer": "str",
    "Address1": "str",
    "Address2": "str",
    "City": "str",
    "State": "str",
    "Zip": "float64",
    "Latitude": "float64",
    "Longitude": "float64",
    "Primary NAICS": "str",
    "Hospitalized": "float64",
    "Amputation": "float64",
    "Loss of Eye": "float64",
    "Inspection": "str",
    "Final Narrative": "str",
    "Nature": "int64",
    "NatureTitle": "str",
    "Part of Body": "int64",
    "Part of Body Title": "str",
    "Event": "int64",
    "EventTitle": "str",
    "Source": "int64",
    "SourceTitle": "str",
    "Secondary Source": "float64",
    "Secondary Source Title": "str",
    "FederalState": "int64",
}


class OshaSIR:
    def __init__(
        self,
        name: str,
        url: str,
        sha256: str | None = None,
        user_agent: str = "Mozilla/5.0 (compatible; research-client/0.1)",
        timeout_s: float = 120.0,
    ) -> None:
        self.name = name
        self.url = url
        self.sha256 = sha256
        self.user_agent = user_agent
        self.timeout_s = timeout_s

    def _manual_help(self, raw_dir: Path, reason: str) -> str:
        pin = f"\n  Expected SHA-256: {self.sha256}" if self.sha256 else ""
        return (
            f"Could not download OSHA Severe Injury Reports: {reason}\n"
            f"  Download the full SIR data set ZIP by hand from {SOURCE_PAGE}\n"
            f"  (pinned URL: {self.url})\n"
            f"  and save it as {raw_dir / ZIP_NAME}, then re-run `recal download`.{pin}\n"
            f"  If OSHA has published a newer file, update url and sha256 in "
            f"conf/dataset/{self.name}.yaml."
        )

    def download(self, raw_dir: Path) -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / ZIP_NAME
        if is_current(raw_dir, target, self.sha256):
            return
        if target.exists() and (self.sha256 is None or sha256_file(target) == self.sha256):
            write_manifest(raw_dir, target, source=f"{self.url} (placed by hand)")
            return
        part = target.with_suffix(".part")
        try:
            headers = {"User-Agent": self.user_agent}
            with (
                httpx.stream(
                    "GET", self.url, headers=headers, timeout=self.timeout_s, follow_redirects=True
                ) as r,
                part.open("wb") as f,
            ):
                r.raise_for_status()
                for chunk in r.iter_bytes():
                    f.write(chunk)
        except httpx.HTTPError as e:
            part.unlink(missing_ok=True)
            raise DownloadError(self._manual_help(raw_dir, f"{type(e).__name__}: {e}")) from e
        actual = sha256_file(part)
        if self.sha256 is not None and actual != self.sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(
                self._manual_help(raw_dir, f"SHA-256 mismatch (got {actual}); the file changed")
            )
        part.replace(target)
        write_manifest(raw_dir, target, source=self.url)

    def load(self, raw_dir: Path) -> pd.DataFrame:
        csv_cols = {c: t for c, t in SCHEMA.items() if c != "id"}
        df = pd.read_csv(raw_dir / ZIP_NAME, dtype=csv_cols, encoding="utf-8")
        df.insert(0, "id", "osha-" + df["UPA"].astype("str"))  # ID is not unique; UPA is
        return apply_schema(df, SCHEMA)

    def schema(self) -> dict[str, str]:
        return dict(SCHEMA)
