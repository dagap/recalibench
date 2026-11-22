"""Label rules: turn dataset columns into a task's ground truth.

A rule gives each row a stratum (used to sample and split before any label
parameter is fitted), is fitted on the fit split only, then labels every row.
NA from `strata` or `apply` means "drop this row". A rule that config can't
express is a class with the same three methods, named by `_target_`.
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from hydra.utils import instantiate


@runtime_checkable
class LabelRule(Protocol):
    def strata(self, df: pd.DataFrame) -> pd.Series: ...
    def fit(self, df: pd.DataFrame) -> None: ...
    def apply(self, df: pd.DataFrame) -> pd.Series: ...
    def params(self) -> dict[str, Any]: ...


def _as_strata(labels: pd.Series) -> pd.Series:
    return labels.astype("string")


class ColumnGt:
    """1 if `column > value`, else 0; NA where the column is missing."""

    def __init__(self, column: str, value: float = 0) -> None:
        self.column = column
        self.value = value

    def strata(self, df: pd.DataFrame) -> pd.Series:
        return _as_strata(self.apply(df))

    def fit(self, df: pd.DataFrame) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.Series:
        col = pd.to_numeric(df[self.column], errors="coerce")
        return (col > self.value).astype("Int64").mask(col.isna())

    def params(self) -> dict[str, Any]:
        return {"rule": "column_gt", "column": self.column, "value": self.value}


class QuantileGt:
    """1 if `column` is above its q-quantile, with the threshold fitted on the fit split.

    Sampling and splitting happen before the threshold exists, so rows are
    stratified on `bins` equal-frequency bins of the raw column instead.
    """

    def __init__(self, column: str, q: float, bins: int = 20) -> None:
        if not 0 < q < 1:
            raise ValueError("q must be in (0, 1)")
        self.column = column
        self.q = q
        self.bins = bins
        self.threshold: float | None = None

    def _col(self, df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df[self.column], errors="coerce")

    def strata(self, df: pd.DataFrame) -> pd.Series:
        col = self._col(df)
        ranks = col.rank(method="first")
        binned = pd.qcut(ranks, q=min(self.bins, int(col.notna().sum())), labels=False)
        return _as_strata(pd.Series(binned, index=df.index).astype("Int64"))

    def fit(self, df: pd.DataFrame) -> None:
        self.threshold = float(self._col(df).quantile(self.q))

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.threshold is None:
            raise RuntimeError("QuantileGt.apply called before fit")
        col = self._col(df)
        return (col > self.threshold).astype("Int64").mask(col.isna())

    def params(self) -> dict[str, Any]:
        return {
            "rule": "quantile_gt",
            "column": self.column,
            "q": self.q,
            "threshold": self.threshold,
        }


class MapValues:
    """Map each value of `column` (as a string) to a label; unmapped values drop."""

    def __init__(self, column: str, mapping: Mapping[str, Any]) -> None:
        self.column = column
        self.mapping = {str(k): v for k, v in mapping.items()}

    def strata(self, df: pd.DataFrame) -> pd.Series:
        return _as_strata(self.apply(df))

    def fit(self, df: pd.DataFrame) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.Series:
        keys = df[self.column].astype("string").str.strip()
        return keys.map(self.mapping).astype("object").where(keys.isin(list(self.mapping)))

    def params(self) -> dict[str, Any]:
        return {"rule": "map_values", "column": self.column, "mapping": self.mapping}


class CodePrefix(MapValues):
    """Map the first `digits` characters of a code, e.g. OIICS event divisions."""

    def __init__(self, column: str, mapping: Mapping[str, Any], digits: int = 1) -> None:
        super().__init__(column, mapping)
        self.digits = digits

    def apply(self, df: pd.DataFrame) -> pd.Series:
        prefix = df[self.column].astype("string").str.strip().str[: self.digits]
        return prefix.map(self.mapping).astype("object").where(prefix.isin(list(self.mapping)))

    def params(self) -> dict[str, Any]:
        return {**super().params(), "rule": "code_prefix", "digits": self.digits}


RULES: dict[str, type] = {
    "column_gt": ColumnGt,
    "quantile_gt": QuantileGt,
    "map_values": MapValues,
    "code_prefix": CodePrefix,
}


def make_rule(cfg: Mapping[str, Any]) -> LabelRule:
    """Build a rule from a task's `label:` block: `rule: <name>` or `_target_: <class>`."""
    kwargs = dict(cfg)
    if "_target_" in kwargs:
        rule: object = instantiate(kwargs, _convert_="all")
    else:
        name = kwargs.pop("rule", None)
        if name not in RULES:
            raise ValueError(f"unknown label rule {name!r}; known: {sorted(RULES)}")
        rule = RULES[str(name)](**kwargs)
    if not isinstance(rule, LabelRule):
        raise TypeError(f"{type(rule).__qualname__} is not a LabelRule")
    return rule
