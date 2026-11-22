from pathlib import Path

import pandas as pd
import pytest

from recalibench.core.config import load_config
from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.core.types import TaskSpec
from recalibench.pipeline.prepare import task_settings
from recalibench.tasks.builder import (
    RATE_TOLERANCE,
    build_frame,
    check_splits,
    rate_spread,
    read_items,
    write,
)
from recalibench.tasks.state import StateSpec

FRACTIONS = {"fit": 0.6, "val": 0.2, "test": 0.2}
ROOT = Path(__file__).resolve().parents[2]
DEMO_TASKS = ["osha_amputation", "osha_event_type", "wc_large_loss"]

# Columns that leak a demo task's label (or the insurer's estimate) into the state.
FORBIDDEN = {
    "osha_sir": {"Amputation", "Nature", "NatureTitle", "Event", "EventTitle"},
    "workers_comp": {"UltimateIncurredClaimCost", "InitialCaseEstimate"},
}


def _spec(**over: object) -> TaskSpec:
    raw: dict[str, object] = {
        "dataset": "d",
        "question": {"kind": "binary", "text": "Positive?"},
        "state": {"fields": ["text"]},
        "label": {"rule": "column_gt", "column": "y", "value": 0},
        **over,
    }
    return TaskSpec.from_dict("t", raw)  # type: ignore[arg-type]


def _raw(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"r{i:04d}" for i in range(n)],
            "text": [f"row {i}" for i in range(n)],
            "y": [int(i % 4 == 0) for i in range(n)],
        }
    )


def test_drops_blank_duplicate_and_unlabelled_rows() -> None:
    raw = pd.concat(
        [
            _raw(100),
            pd.DataFrame(
                {"id": ["z1", "z2", "z3"], "text": ["", "row 1", "fresh"], "y": [1, 1, None]}
            ),
        ]
    )
    built = build_frame(_spec(), raw, seed=0, sample_size=1000, fractions=FRACTIONS)
    assert built.dropped == {
        "empty_state": 1,
        "duplicate_state": 1,
        "no_label": 1,
        "unlabelled_after_fit": 0,
    }
    assert len(built.frame) == 100
    assert "r0001" in set(built.frame["id"])  # the first row by id wins the dedupe


def test_splits_disjoint_and_rates_balanced() -> None:
    built = build_frame(_spec(), _raw(), seed=0, sample_size=300, fractions=FRACTIONS)
    assert len(built.frame) == 300
    assert check_splits(built.frame, _spec(), list(FRACTIONS)) == []
    assert max(rate_spread(built.frame).values()) <= RATE_TOLERANCE


def test_same_seed_same_frame_other_seed_differs() -> None:
    a = build_frame(_spec(), _raw(), seed=7, sample_size=200, fractions=FRACTIONS).frame
    b = build_frame(_spec(), _raw(), seed=7, sample_size=200, fractions=FRACTIONS).frame
    c = build_frame(_spec(), _raw(), seed=8, sample_size=200, fractions=FRACTIONS).frame
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_check_splits_catches_leaks_and_bad_labels() -> None:
    frame = pd.DataFrame(
        {"id": ["a", "a"], "state_json": ["{}", "{}"], "label": [0, 5], "split": ["fit", "val"]}
    )
    problems = " | ".join(check_splits(frame, _spec(), list(FRACTIONS)))
    assert "id appears more than once" in problems
    assert "identical state" in problems
    assert "labels outside" in problems
    assert "split 'test' is empty" in problems


def test_choice_labels_must_be_option_keys() -> None:
    spec = _spec(
        question={
            "kind": "choice",
            "text": "Which?",
            "options": [{"key": "a", "label": "A"}, {"key": "b", "label": "B"}],
        },
        label={"rule": "map_values", "column": "y", "mapping": {"0": "a", "1": "c"}},
    )
    built = build_frame(spec, _raw(), seed=0, sample_size=100, fractions=FRACTIONS)
    assert any("labels outside" in p for p in check_splits(built.frame, spec, list(FRACTIONS)))


def test_write_and_read_items_roundtrip(tmp_path: Path) -> None:
    built = build_frame(_spec(), _raw(), seed=0, sample_size=100, fractions=FRACTIONS)
    paths = Paths(tmp_path, "mock")
    write(built, _spec(), paths, "fp", {})
    items = read_items(paths, "t")
    assert {k: len(v) for k, v in items.items()} == {"fit": 60, "val": 20, "test": 20}
    first = items["fit"][0]
    assert isinstance(first.label, int) and set(first.state) == {"text"}


@pytest.mark.parametrize("task", DEMO_TASKS)
def test_demo_task_state_never_leaks_label(task: str) -> None:
    spec = Registry().task(task)
    cols = set(StateSpec.parse(spec.state).columns)
    assert not cols & FORBIDDEN[spec.dataset]
    assert spec.label["column"] not in cols


@pytest.mark.parametrize("task", DEMO_TASKS)
def test_real_demo_tasks_meet_split_checks(task: str) -> None:
    reg, cfg = Registry(), load_config()
    spec = reg.task(task)
    raw_dir = Paths(ROOT, "mock").raw_dir(spec.dataset)
    if not raw_dir.is_dir():
        pytest.skip(f"raw data for {spec.dataset} not downloaded")
    sample_size, fractions = task_settings(spec, cfg)
    built = build_frame(
        spec,
        reg.dataset(spec.dataset).load(raw_dir),
        seed=42,
        sample_size=sample_size,
        fractions=fractions,
    )
    assert len(built.frame) == sample_size
    assert check_splits(built.frame, spec, list(fractions)) == []
    assert max(rate_spread(built.frame).values()) <= RATE_TOLERANCE
