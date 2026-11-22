import pytest

from recalibench.core.types import Option, Question, TaskSpec


def test_binary_question_keys() -> None:
    assert Question(kind="binary", text="Yes?").keys == ("yes", "no")


def test_choice_question_keys_follow_options() -> None:
    q = Question(kind="choice", text="Which?", options=(Option("a", "A"), Option("b", "B")))
    assert q.keys == ("a", "b")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "score", "text": "x"},
        {"kind": "binary", "text": "  "},
        {"kind": "binary", "text": "x", "options": (Option("a", "A"),)},
        {"kind": "choice", "text": "x", "options": (Option("a", "A"),)},
        {"kind": "choice", "text": "x", "options": (Option("a", "A"), Option("a", "B"))},
    ],
)
def test_invalid_questions_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Question(**kwargs)  # type: ignore[arg-type]


def test_taskspec_from_dict_parses_choice_and_keeps_extras() -> None:
    spec = TaskSpec.from_dict(
        "t",
        {
            "dataset": "d",
            "question": {
                "kind": "choice",
                "text": "Which?",
                "options": [{"key": "a", "label": "A"}, {"key": "b", "label": "B"}],
            },
            "state": {"fields": ["x"]},
            "label": {"rule": "map_values"},
            "notes": "kept",
        },
    )
    assert spec.question.keys == ("a", "b")
    assert spec.extra == {"notes": "kept"}
    assert spec.sample_size is None


def test_taskspec_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing"):
        TaskSpec.from_dict("t", {"dataset": "d"})
