import numpy as np
import pandas as pd
import pytest

from recalibench.tasks.label_rules import (
    CodePrefix,
    ColumnGt,
    LabelRule,
    MapValues,
    QuantileGt,
    make_rule,
)


def test_column_gt_marks_missing_as_na() -> None:
    out = ColumnGt("x", 0).apply(pd.DataFrame({"x": [0.0, 1.0, 2.0, np.nan]}))
    assert out.tolist()[:3] == [0, 1, 1]
    assert pd.isna(out.iloc[3])


def test_quantile_threshold_uses_only_the_fit_rows() -> None:
    fit = pd.DataFrame({"cost": np.arange(1.0, 101.0)})  # q90 = 90.1
    rest = pd.DataFrame({"cost": [1e9, 1e9, 0.0]})  # extreme val/test values
    rule = QuantileGt("cost", q=0.9)
    rule.fit(fit)
    before = rule.threshold
    assert before == pytest.approx(90.1)
    assert rule.apply(rest).tolist() == [1, 1, 0]
    assert rule.threshold == before


def test_quantile_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        QuantileGt("cost", 0.9).apply(pd.DataFrame({"cost": [1.0]}))


def test_quantile_strata_are_equal_frequency_bins() -> None:
    strata = QuantileGt("cost", 0.9, bins=20).strata(pd.DataFrame({"cost": np.arange(200.0)}))
    assert strata.value_counts().tolist() == [10] * 20


def test_map_values_drops_unmapped() -> None:
    out = MapValues("g", {"a": "x", "b": "y"}).apply(pd.DataFrame({"g": ["a", "b", "c"]}))
    assert out.tolist()[:2] == ["x", "y"] and pd.isna(out.iloc[2])


def test_code_prefix_maps_divisions_and_drops_nonclassifiable() -> None:
    rule = CodePrefix("Event", {"4": "fall", "6": "contact"}, digits=1)
    out = rule.apply(pd.DataFrame({"Event": [4330, 642, 9999]}))
    assert out.tolist()[:2] == ["fall", "contact"] and pd.isna(out.iloc[2])


def test_make_rule_by_name_and_by_target() -> None:
    assert isinstance(make_rule({"rule": "column_gt", "column": "x"}), ColumnGt)
    target = {"_target_": "recalibench.tasks.label_rules.ColumnGt", "column": "x", "value": 2}
    rule = make_rule(target)
    assert isinstance(rule, LabelRule) and rule.params()["value"] == 2


def test_make_rule_unknown() -> None:
    with pytest.raises(ValueError, match="unknown label rule"):
        make_rule({"rule": "nope"})
