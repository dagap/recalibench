"""Build what a model sees for each row, from a task's `state:` block.

    state:
      fields: [ClaimDescription, Age]          # or {alias: column, ...}
      values: {Gender: {M: male, F: female}}   # optional recoding, keyed by alias

State JSON keeps the configured key order and is compact, so identical states
serialise identically (dedupe and cache keys depend on that).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateSpec:
    fields: dict[str, str]  # alias -> column
    values: dict[str, dict[str, str]]  # alias -> {raw value: shown value}

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "StateSpec":
        fields_raw = raw.get("fields")
        if isinstance(fields_raw, list):
            fields = {str(c): str(c) for c in cast(list[Any], fields_raw)}
        elif isinstance(fields_raw, Mapping):
            fields = {str(k): str(v) for k, v in cast(Mapping[Any, Any], fields_raw).items()}
        else:
            raise ValueError("state.fields must be a list of columns or an {alias: column} map")
        if not fields:
            raise ValueError("state.fields is empty")
        values_raw = cast(Mapping[Any, Any], raw.get("values") or {})
        values = {
            str(alias): {str(k): str(v) for k, v in cast(Mapping[Any, Any], m).items()}
            for alias, m in values_raw.items()
        }
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"state.values names unknown fields {sorted(unknown)}")
        extra = set(raw) - {"fields", "values"}
        if extra:
            raise ValueError(f"unknown state keys {sorted(extra)}")
        return cls(fields=fields, values=values)

    @property
    def columns(self) -> list[str]:
        return list(self.fields.values())


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return cast(Any, value.item())  # pyright: ignore[reportUnknownMemberType]
    return value


def _blank(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def build_states(df: pd.DataFrame, spec: StateSpec) -> pd.Series:
    """State JSON per row; NA where any state field is null or blank."""
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise ValueError(f"state columns not in dataset: {missing}")
    out: list[str | None] = []
    for row in df[spec.columns].itertuples(index=False, name=None):
        state: dict[str, Any] = {}
        ok = True
        for alias, value in zip(spec.fields, row, strict=True):
            if _blank(value):
                ok = False
                break
            value = _native(value)
            if isinstance(value, str):
                value = value.strip()
            recode = spec.values.get(alias)
            if recode is not None:
                value = recode.get(str(value), value)
            state[alias] = value
        out.append(json.dumps(state, ensure_ascii=False, separators=(",", ":")) if ok else None)
    return pd.Series(out, index=df.index, dtype="string")
