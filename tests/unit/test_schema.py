import pandas as pd
import pytest

from recalibench.core.schema import apply_schema, check_schema

SCHEMA = {"id": "str", "x": "int64"}


def test_clean_frame_passes() -> None:
    df = apply_schema(pd.DataFrame({"id": ["a", "b"], "x": [1, 2], "extra": [0, 0]}), SCHEMA)
    assert list(df.columns) == ["id", "x"]
    assert check_schema(df, SCHEMA) == []


def test_reports_every_problem() -> None:
    df = pd.DataFrame({"id": ["a", "a"], "x": [1.5, 2.0]})
    problems = check_schema(df, {**SCHEMA, "y": "str"})
    assert "column 'x' is float64, expected int64" in problems
    assert "missing column 'y'" in problems
    assert "column 'id' is not unique" in problems


def test_missing_id_reported() -> None:
    assert "missing column 'id'" in check_schema(pd.DataFrame({"x": [1]}), {"x": "int64"})


def test_apply_schema_missing_column() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        apply_schema(pd.DataFrame({"id": ["a"]}), SCHEMA)
