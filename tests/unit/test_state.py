import json

import numpy as np
import pandas as pd
import pytest

from recalibench.tasks.state import StateSpec, build_states


def test_list_fields_keep_order_and_native_types() -> None:
    df = pd.DataFrame({"b": [2], "a": [" hi "]})
    [s] = build_states(df, StateSpec.parse({"fields": ["b", "a"]})).tolist()
    assert s == '{"b":2,"a":"hi"}'


def test_aliases_and_value_recoding() -> None:
    spec = StateSpec.parse(
        {"fields": {"pt": "PartTimeFullTime", "g": "Gender"}, "values": {"pt": {"F": "full-time"}}}
    )
    [s] = build_states(pd.DataFrame({"PartTimeFullTime": ["F"], "Gender": ["F"]}), spec).tolist()
    assert json.loads(s) == {"pt": "full-time", "g": "F"}  # recoding applies per field only


@pytest.mark.parametrize("bad", [None, np.nan, "", "   "])
def test_null_or_blank_field_drops_row(bad: object) -> None:
    out = build_states(pd.DataFrame({"t": ["ok", bad]}), StateSpec.parse({"fields": ["t"]}))
    assert out.iloc[0] == '{"t":"ok"}' and pd.isna(out.iloc[1])


@pytest.mark.parametrize(
    "raw",
    [
        {"fields": []},
        {"fields": "t"},
        {"fields": ["t"], "values": {"x": {}}},
        {"fields": ["t"], "z": 1},
    ],
)
def test_invalid_specs(raw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StateSpec.parse(raw)


def test_missing_column() -> None:
    with pytest.raises(ValueError, match="not in dataset"):
        build_states(pd.DataFrame({"a": [1]}), StateSpec.parse({"fields": ["b"]}))
