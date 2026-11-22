"""Synthetic workers' compensation claims, OpenML data_id 42876 (100,000 rows).

fetch_openml drops ClaimNumber (OpenML marks it as an ignored attribute), so
ids are the row position, which is stable because data_id pins the version.
"""

from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.datasets import (  # pyright: ignore[reportMissingTypeStubs]
    fetch_openml,  # pyright: ignore[reportUnknownVariableType]
)

from recalibench.core.schema import apply_schema
from recalibench.datasets.manifest import DownloadError, is_current, write_manifest

PARQUET_NAME = "workers_comp.parquet"

SCHEMA: dict[str, str] = {
    "id": "str",
    "DateTimeOfAccident": "str",
    "DateReported": "str",
    "Age": "int64",
    "Gender": "str",
    "MaritalStatus": "str",
    "DependentChildren": "int64",
    "DependentsOther": "int64",
    "WeeklyPay": "int64",
    "PartTimeFullTime": "str",
    "HoursWorkedPerWeek": "int64",
    "DaysWorkedPerWeek": "int64",
    "ClaimDescription": "str",
    "InitialCaseEstimate": "float64",  # the insurer's estimate: never put it in a task state
    "UltimateIncurredClaimCost": "float64",
}


def _fetch(openml_id: int, data_home: Path) -> pd.DataFrame:
    bunch: Any = fetch_openml(  # pyright: ignore[reportUnknownVariableType]
        data_id=openml_id, as_frame=True, parser="pandas", data_home=str(data_home)
    )
    return cast(pd.DataFrame, bunch.frame)


class WorkersComp:
    def __init__(self, name: str, openml_id: int = 42876) -> None:
        self.name = name
        self.openml_id = openml_id

    def download(self, raw_dir: Path) -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / PARQUET_NAME
        if is_current(raw_dir, target):
            return
        try:
            df = _fetch(self.openml_id, raw_dir / "openml")
        except Exception as e:  # network, HTTP and ARFF parse errors all land here
            raise DownloadError(
                f"Could not fetch OpenML dataset {self.openml_id}: {type(e).__name__}: {e}\n"
                f"  Check your network and retry; OpenML needs no login."
            ) from e
        df = df.reset_index(drop=True)
        df.insert(0, "id", [f"wc-{i:06d}" for i in range(len(df))])
        apply_schema(df, SCHEMA).to_parquet(target, index=False)
        write_manifest(
            raw_dir, target, source=f"https://www.openml.org/d/{self.openml_id}", rows=len(df)
        )

    def load(self, raw_dir: Path) -> pd.DataFrame:
        return apply_schema(pd.read_parquet(raw_dir / PARQUET_NAME), SCHEMA)

    def schema(self) -> dict[str, str]:
        return dict(SCHEMA)
