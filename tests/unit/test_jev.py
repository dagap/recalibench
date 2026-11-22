"""Jev adapter: wire format per docs.typesafe.ai (recorded examples from the docs)."""

import pytest

from recalibench.core.types import Item, Option, Question
from recalibench.models.base_api import FatalApiError, ParseError, validate_probs
from recalibench.models.jev import JevModel

BINARY = Question(kind="binary", text="Is this message about billing?")
TONE = Question(
    kind="choice",
    text="What is the tone of this message?",
    options=(
        Option("angry", "angry", "An upset or hostile message"),
        Option("calm", "calm", "A neutral or polite message"),
        Option("excited", "excited", "An enthusiastic or eager message"),
    ),
)
ITEM = Item(
    id="m1",
    state={"message": "I was charged twice for my subscription this month."},
    label=1,
)


def _jev() -> JevModel:
    return JevModel(
        name="jev",
        version="jev-1.13.0",
        base_url="https://api.typesafe.ai/v1/",
        api_key_env="TYPESAFE_API_KEY",
        price_per_m_input_tokens=0.042,
    )


def test_endpoint() -> None:
    assert _jev().endpoint == "https://api.typesafe.ai/v1/systemone"


def test_binary_request_shape() -> None:
    assert _jev().build_request(ITEM, {"q": BINARY}) == {
        "model": "jev-1.13.0",
        "state": ITEM.state,
        "questions": {"q": {"type": "noul", "instructions": "Is this message about billing?"}},
    }


def test_choice_request_uses_labels_as_criteria() -> None:
    q = Question(
        kind="choice",
        text="Kind?",
        options=(Option("fall", "Falls, slips, trips", "A fall"), Option("other", "Other")),
    )
    body = _jev().build_request(ITEM, {"q": q})
    assert body["questions"]["q"] == {
        "type": "choice",
        "instructions": "Kind?",
        "criteria": {"Falls, slips, trips": "A fall", "Other": None},
    }


def test_parse_docs_noul_example() -> None:
    raw = {
        "model": "jev-1.13.0",
        "answers": {"q": {"type": "noul", "noul": 0.98}},
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }
    parsed, usage = _jev().parse_response(raw, {"q": BINARY})
    probs = parsed["q"]
    assert probs == pytest.approx({"yes": 0.98, "no": 0.02})
    assert (usage.input_tokens, usage.output_tokens) == (120, 12)
    assert _jev().price(usage) == pytest.approx(120 * 0.042 / 1e6)


def test_parse_docs_choice_example_maps_labels_to_keys() -> None:
    q = Question(
        kind="choice",
        text="Tone?",
        options=(Option("a", "angry"), Option("c", "calm"), Option("e", "excited")),
    )
    raw = {
        "model": "jev-1.13.0",
        "answers": {
            "q": {
                "type": "choice",
                "choice": "calm",
                "confidence": 0.7,
                "probabilities": {"angry": 0.15, "calm": 0.8, "excited": 0.05},
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }
    probs = _jev().parse_response(raw, {"q": q})[0]["q"]
    assert probs == {"a": 0.15, "c": 0.8, "e": 0.05}  # confidence (0.7) is ignored


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"model": "jev-1.14.0", "answers": {}, "usage": {}}, "pinned"),
        ({"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 1}}, "KeyError"),
        (
            {"model": "jev-1.13.0", "answers": {"q": {"noul": "x"}}, "usage": {"input_tokens": 1}},
            "ValueError",
        ),
        ({"model": "jev-1.13.0", "answers": {"q": {"noul": 0.5}}}, "KeyError"),
    ],
)
def test_bad_replies_raise_parse_error(raw: dict[str, object], match: str) -> None:
    with pytest.raises(ParseError, match=match):
        _jev().parse_response(raw, {"q": BINARY})


def test_unknown_choice_option_rejected() -> None:
    raw = {
        "model": "jev-1.13.0",
        "answers": {"q": {"probabilities": {"angry": 0.5, "sad": 0.5}}},
        "usage": {"input_tokens": 1},
    }
    with pytest.raises(ParseError, match="unknown options"):
        _jev().parse_response(raw, {"q": TONE})


def test_missing_key_is_fatal_not_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(FatalApiError, match="TYPESAFE_API_KEY"):
        _jev().headers()
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    assert _jev().headers() == {"Authorization": "Bearer sk-test"}


def test_fake_response_roundtrips_through_parse() -> None:
    jev = _jev()
    for q, probs in [
        (BINARY, {"yes": 0.3, "no": 0.7}),
        (TONE, {"angry": 0.2, "calm": 0.5, "excited": 0.3}),
    ]:
        request = jev.build_request(ITEM, {"q": q})
        raw = jev.fake_response(request, {"q": q}, {"q": probs})
        parsed, usage = jev.parse_response(raw, {"q": q})
        assert validate_probs(parsed["q"], q) == pytest.approx(probs)
        assert usage.input_tokens == jev.count_tokens(request)


def test_duplicate_choice_labels_rejected() -> None:
    q = Question(kind="choice", text="?", options=(Option("a", "Same"), Option("b", "Same")))
    with pytest.raises(ValueError, match="unique labels"):
        _jev().build_request(ITEM, {"q": q})


@pytest.mark.parametrize(
    "probs",
    [
        {"yes": 0.5},
        {"yes": 1.5, "no": -0.5},
        {"yes": 0.5, "no": 0.6},
        {"yes": float("nan"), "no": 0},
    ],
)
def test_validate_probs_rejects(probs: dict[str, float]) -> None:
    with pytest.raises(ParseError):
        validate_probs(probs, BINARY)


def test_several_named_questions_in_one_request() -> None:
    jev = _jev()
    qs = {"a": BINARY, "b": TONE}
    body = jev.build_request(ITEM, qs)
    assert set(body["questions"]) == {"a", "b"}
    assert body["questions"]["b"]["type"] == "choice"
    raw = jev.fake_response(
        body, qs, {"a": {"yes": 0.9, "no": 0.1}, "b": {"angry": 0.1, "calm": 0.6, "excited": 0.3}}
    )
    parsed, _ = jev.parse_response(raw, qs)
    assert parsed["a"]["yes"] == pytest.approx(0.9) and parsed["b"]["calm"] == pytest.approx(0.6)


def test_missing_named_answer_is_a_parse_error() -> None:
    raw = {"model": "jev-1.13.0", "answers": {"a": {"noul": 0.5}}, "usage": {"input_tokens": 1}}
    with pytest.raises(ParseError, match="KeyError"):
        _jev().parse_response(raw, {"a": BINARY, "b": BINARY})
