"""Phase 4 check: the mock model is miscalibrated by exactly the intended amount."""

import numpy as np
import pytest

from recalibench.core.types import Item, Option, Question
from recalibench.models.mock_generator import MockGenerator
from tests.helpers.calibration import binary_ece, ece

BINARY = Question(kind="binary", text="Positive?")
CHOICE = Question(kind="choice", text="Which?", options=tuple(Option(k, k.upper()) for k in "abcd"))
N = 5000


def _items(question: Question, n: int = N) -> list[Item]:
    rng = np.random.default_rng(0)
    keys = question.keys if question.kind == "choice" else None
    out: list[Item] = []
    for i in range(n):
        label: int | str = str(rng.choice(np.array(keys))) if keys else int(rng.integers(0, 2))
        out.append(Item(id=f"i{i}", state={"n": i}, label=label))
    return out


def _top(probs: list[dict[str, float]], items: list[Item]) -> tuple[list[float], list[int]]:
    conf = [max(p.values()) for p in probs]
    hit = [int(max(p, key=lambda k: p[k]) == it.label) for p, it in zip(probs, items, strict=True)]
    return conf, hit


def test_binary_is_miscalibrated_and_under_confident() -> None:
    gen, items = MockGenerator(), _items(BINARY)
    p = [gen.probs("jev", it, BINARY, 0)["yes"] for it in items]
    labels = [int(it.label) for it in items]
    assert binary_ece(p, labels) > 0.05
    # Under-confident: among confident "yes" answers, the truth is even more often yes.
    hi = [lab for pi, lab in zip(p, labels, strict=True) if pi > 0.7]
    assert np.mean(hi) > np.mean([pi for pi in p if pi > 0.7])


def test_choice_is_miscalibrated_and_over_confident() -> None:
    gen, items = MockGenerator(), _items(CHOICE)
    conf, hit = _top([gen.probs("jev", it, CHOICE, 0) for it in items], items)
    assert ece(conf, hit) > 0.05
    assert np.mean(conf) > np.mean(hit)  # says more than it knows


def test_undoing_the_known_distortion_calibrates_it() -> None:
    gen = MockGenerator(noise_sd=0.0)
    items = _items(BINARY)
    p = [1 / (1 + np.exp(-float(gen.calibrated_logits("jev", it, BINARY)[0]))) for it in items]
    assert binary_ece(p, [int(it.label) for it in items]) < 0.02

    items = _items(CHOICE)
    probs: list[dict[str, float]] = []
    for it in items:
        z = gen.calibrated_logits("jev", it, CHOICE)
        e = np.exp(z - z.max())
        probs.append(dict(zip(CHOICE.keys, (e / e.sum()).tolist(), strict=True)))
    assert ece(*_top(probs, items)) < 0.02


def test_deterministic_and_repeats_differ_slightly() -> None:
    gen, item = MockGenerator(), Item(id="x", state={"t": "hi"}, label=1)
    assert gen.probs("jev", item, BINARY, 0) == gen.probs("jev", item, BINARY, 0)
    a, b = gen.probs("jev", item, BINARY, 0)["yes"], gen.probs("jev", item, BINARY, 1)["yes"]
    assert a != b and abs(a - b) < 0.1
    assert gen.probs("other", item, BINARY, 0) != gen.probs("jev", item, BINARY, 0)


def test_choice_label_must_be_an_option() -> None:
    with pytest.raises(ValueError, match="not an option key"):
        MockGenerator().probs("jev", Item(id="x", state={}, label="zzz"), CHOICE, 0)
