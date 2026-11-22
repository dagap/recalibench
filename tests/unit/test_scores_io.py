from pathlib import Path

import pytest

from recalibench.core.types import Item, Option, Prediction, Question
from recalibench.pipeline.scores_io import CLIP, build_frame, clip_probs, read_scores, write_scores

BINARY = Question(kind="binary", text="?")
CHOICE = Question(
    kind="choice", text="?", options=(Option("a", "A"), Option("b", "B"), Option("c", "C"))
)


def _pred(item_id: str, probs: dict[str, float], std: dict[str, float] | None = None) -> Prediction:
    return Prediction(
        item_id=item_id, probs=probs, std=std, model="m@1", latency_ms=5.0, cost_usd=0.1
    )


def test_binary_clip_is_exact() -> None:
    assert clip_probs([1.0, 0.0], "binary") == pytest.approx([1 - CLIP, CLIP], abs=1e-15)
    assert clip_probs([0.3, 0.7], "binary") == pytest.approx([0.3, 0.7])


def test_choice_clip_renormalises() -> None:
    p = clip_probs([1.0, 0.0, 0.0], "choice")
    assert sum(p) == pytest.approx(1.0) and min(p) > 0


def test_frame_skips_missing_predictions_and_roundtrips(tmp_path: Path) -> None:
    items = [Item("x", {}, "a"), Item("y", {}, "b"), Item("z", {}, "c")]
    preds = {
        "x": _pred("x", {"a": 0.7, "b": 0.2, "c": 0.1}, {"a": 0.01, "b": 0.0, "c": 0.0}),
        "z": _pred("z", {"a": 0.0, "b": 0.0, "c": 1.0}),
    }
    frame = build_frame(items, ["fit", "val", "test"], preds, CHOICE)
    assert frame["id"].tolist() == ["x", "z"]
    assert list(frame.columns) == [
        "id",
        "split",
        "label",
        "p_a",
        "p_b",
        "p_c",
        "std_a",
        "std_b",
        "std_c",
        "latency_ms",
        "cost_usd",
    ]
    assert bool(frame["std_a"].isna().iloc[1])
    write_scores(tmp_path, frame, {"keys": ["a", "b", "c"], "rows": 2})
    scores = read_scores(tmp_path)
    assert scores.probs().shape == (2, 3)
    assert scores.probs("test")[0] == pytest.approx(
        [CLIP / (1 + 2 * CLIP)] * 2 + [1 / (1 + 2 * CLIP)]
    )
    assert scores.labels("fit").tolist() == ["a"]


def test_binary_labels_are_ints() -> None:
    frame = build_frame(
        [Item("x", {}, 1)], ["fit"], {"x": _pred("x", {"yes": 0.9, "no": 0.1})}, BINARY
    )
    assert str(frame["label"].dtype) == "int64"
    assert frame[["p_yes", "p_no"]].iloc[0].tolist() == pytest.approx([0.9, 0.1])


def test_unsafe_option_key_rejected() -> None:
    q = Question(kind="choice", text="?", options=(Option("a b", "A"), Option("c", "C")))
    with pytest.raises(ValueError, match="column name"):
        build_frame([], [], {}, q)


def test_read_scores_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_scores(tmp_path)
