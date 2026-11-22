import asyncio
from collections import Counter

import numpy as np
import pytest

from recalibench.core.protocols import Model, UnsupportedTask
from recalibench.core.types import Item, Option, Question
from recalibench.models.tfidf_logreg import TfidfLogReg, field_roles, n_folds_for, stratified_folds

BINARY = Question(kind="binary", text="Large loss?")
CHOICE = Question(
    kind="choice", text="Which?", options=tuple(Option(k, k) for k in ("a", "b", "c"))
)
WORDS = ["slipped", "ladder", "fell", "cut", "finger", "machine", "burn", "hot", "oil", "lifted"]


def _wc_items(n: int = 300) -> list[Item]:
    rng = np.random.default_rng(0)
    out: list[Item] = []
    for i in range(n):
        y = int(rng.random() < 0.3)
        words = " ".join(rng.choice(WORDS, 4)) + (" ladder fell" if y else " cut")
        state = {
            "ClaimDescription": f"{words} {i}",
            "Age": int(rng.integers(18, 65)),
            "WeeklyPay": float(rng.normal(500 + 300 * y, 50)),
            "Gender": str(rng.choice(["male", "female"])),
        }
        out.append(Item(id=f"w{i}", state=state, label=y))
    return out


def _run(model: TfidfLogReg, items: list[Item], q: Question) -> list[dict[str, float]]:
    return [p.probs for p in asyncio.run(model.predict(items, q))]


def test_field_roles() -> None:
    assert field_roles(_wc_items()) == {
        "ClaimDescription": "text",
        "Age": "numeric",
        "WeeklyPay": "numeric",
        "Gender": "categorical",
    }


def test_satisfies_model_protocol() -> None:
    m: Model = TfidfLogReg(name="tfidf_logreg", version="1")
    assert isinstance(m, Model) and m.trainable


@pytest.mark.parametrize("features", ["text", "structured"])
def test_learns_signal_on_held_out_items(features: str) -> None:
    items = _wc_items()
    model = TfidfLogReg(name="m", version="1", features=features)
    model.fit(items[:200], BINARY)
    probs = _run(model, items[200:], BINARY)
    p_yes = np.array([p["yes"] for p in probs])
    y = np.array([it.label for it in items[200:]])
    assert p_yes[y == 1].mean() - p_yes[y == 0].mean() > 0.3
    assert all(p["yes"] + p["no"] == pytest.approx(1.0) for p in probs)


def test_structured_unsupported_on_text_only_state() -> None:
    items = [Item(f"i{i}", {"narrative": f"case {i} words here"}, i % 2) for i in range(20)]
    with pytest.raises(UnsupportedTask, match="no non-text fields"):
        TfidfLogReg(name="s", version="1", features="structured").fit(items, BINARY)


def test_text_unsupported_without_text_fields() -> None:
    items = [Item(f"i{i}", {"age": i}, i % 2) for i in range(20)]
    with pytest.raises(UnsupportedTask, match="no text fields"):
        TfidfLogReg(name="t", version="1").fit(items, BINARY)


def test_choice_class_missing_from_training_gets_zero_then_sums_to_one() -> None:
    items = [
        Item(f"i{i}", {"t": f"{'apple' if i % 2 else 'pear'} item {i}"}, "ab"[i % 2])
        for i in range(60)
    ]
    model = TfidfLogReg(name="t", version="1", min_df=1)
    model.fit(items, CHOICE)
    for p in _run(model, items[:5], CHOICE):
        assert set(p) == {"a", "b", "c"} and p["c"] == 0.0
        assert sum(p.values()) == pytest.approx(1.0)


def test_single_class_fold_predicts_constant() -> None:
    items = [Item(f"i{i}", {"t": f"text number {i}"}, 1) for i in range(10)]
    model = TfidfLogReg(name="t", version="1")
    model.fit(items, BINARY)
    assert _run(model, items[:2], BINARY) == [{"yes": 1.0, "no": 0.0}] * 2


def test_predict_before_fit() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        _run(TfidfLogReg(name="t", version="1"), _wc_items(2), BINARY)


def test_stratified_folds_balance_labels() -> None:
    labels = [1] * 30 + [0] * 70
    folds = stratified_folds(labels, 5, seed=0)
    for f in range(5):
        got = Counter(lab for lab, g in zip(labels, folds, strict=True) if g == f)
        assert got == {1: 6, 0: 14}
    assert stratified_folds(labels, 5, seed=0) == folds
    assert n_folds_for(1800, 5) == 5 and n_folds_for(6, 5) == 3
