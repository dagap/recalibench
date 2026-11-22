from pathlib import Path

from recalibench.core.types import Option, Question
from recalibench.models.cache import ResponseCache, cache_key

Q = Question(kind="binary", text="Yes?")


def test_key_ignores_state_key_order() -> None:
    assert cache_key({"a": 1, "b": 2}, Q, 0) == cache_key({"b": 2, "a": 1}, Q, 0)


def test_key_changes_with_anything_that_changes_the_answer() -> None:
    base = cache_key({"a": 1}, Q, 0)
    choice = Question(kind="choice", text="Yes?", options=(Option("x", "X"), Option("y", "Y")))
    others = {
        cache_key({"a": 2}, Q, 0),
        cache_key({"a": 1}, Question(kind="binary", text="Really?"), 0),
        cache_key({"a": 1}, Q, 1),
        cache_key({"a": 1}, Q, 0, namespace="mock:..."),
        cache_key({"a": 1}, choice, 0),
    }
    assert base not in others and len(others) == 5


def test_roundtrip_and_persistence(tmp_path: Path) -> None:
    c = ResponseCache(tmp_path / "c")
    c.set("k", {"probs": {"yes": 0.7, "no": 0.3}, "raw": {"x": [1, "é"]}})
    c.close()
    again = ResponseCache(tmp_path / "c")
    assert again.get("k") == {"probs": {"yes": 0.7, "no": 0.3}, "raw": {"x": [1, "é"]}}
    assert again.get("missing") is None and len(again) == 1
    again.close()
