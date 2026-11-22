"""Rebuild the 50-row dataset fixtures from the downloaded raw data.

    uv run recal download --dataset osha_sir --dataset workers_comp
    uv run python tests/fixtures/make_fixtures.py

OSHA rows are anonymised: employer, record numbers (UPA, ID), street address, city,
ZIP and coordinates are replaced. Narrative, date and state are kept because the tasks
need them; the narrative text is still searchable in the public OSHA file.
The sample is stratified so every OIICS event division and both amputation labels appear.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd

from recalibench.datasets.osha import SCHEMA as OSHA_SCHEMA
from recalibench.datasets.workers_comp import PARQUET_NAME

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
N = 50
SEED = 42
OSHA_CSV_NAME = "January2015toNovember2025.csv"
REDACT = ["Address1", "Address2", "City", "Zip", "Latitude", "Longitude"]


def osha() -> None:
    raw = pd.read_csv(ROOT / "data/raw/osha_sir/sir.zip", dtype=str, keep_default_na=False)
    division = raw["Event"].str[0]
    picks = [g.sample(n=min(len(g), 4), random_state=SEED) for _, g in raw.groupby(division)]
    picks.append(raw[raw["Amputation"] == ""].head(1))  # keep a missing-label row
    sample = pd.concat(picks)
    rest = raw.drop(sample.index)
    sample = pd.concat([sample, rest.sample(n=N - len(sample), random_state=SEED)])
    sample = sample.sort_values("UPA").reset_index(drop=True)
    sample["Employer"] = [f"Employer {i:03d}" for i in range(len(sample))]
    # Real UPA/ID numbers would let anyone look the row up in the public file.
    sample["UPA"] = [str(900000 + i) for i in range(len(sample))]
    sample["ID"] = [str(800000 + i) for i in range(len(sample))]
    sample[REDACT] = ""
    assert list(sample.columns) == [c for c in OSHA_SCHEMA if c != "id"]
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    (OUT / "osha_sir").mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "osha_sir/sir.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(OSHA_CSV_NAME, buf.getvalue())


def workers_comp() -> None:
    df = pd.read_parquet(ROOT / "data/raw/workers_comp" / PARQUET_NAME)
    big = df["UltimateIncurredClaimCost"] > df["UltimateIncurredClaimCost"].quantile(0.9)
    sample = pd.concat(
        [df[big].sample(n=10, random_state=SEED), df[~big].sample(n=N - 10, random_state=SEED)]
    ).sort_values("id")
    (OUT / "workers_comp").mkdir(exist_ok=True)
    sample.to_parquet(OUT / "workers_comp" / PARQUET_NAME, index=False)


if __name__ == "__main__":
    osha()
    workers_comp()
    print("fixtures written under", OUT)
