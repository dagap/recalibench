"""Check a loaded frame against a dataset's declared schema."""

import pandas as pd


def apply_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Keep the schema's columns, in order, cast to its dtypes."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df[list(schema)].astype(schema)


def check_schema(df: pd.DataFrame, schema: dict[str, str]) -> list[str]:
    """Every way `df` breaks the contract: schema columns and dtypes, unique string `id`."""
    problems: list[str] = []
    for col, dtype in schema.items():
        if col not in df.columns:
            problems.append(f"missing column {col!r}")
        elif str(df[col].dtype) != dtype:
            problems.append(f"column {col!r} is {df[col].dtype}, expected {dtype}")
    if "id" not in df.columns:
        problems.append("missing column 'id'")
    else:
        if df["id"].isna().any():
            problems.append("column 'id' has nulls")
        if not df["id"].is_unique:
            problems.append("column 'id' is not unique")
    return problems
