"""The mock model is miscalibrated by exactly the intended amount."""

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


def test_first_option_bias_follows_whichever_option_is_listed_first() -> None:
    gen = MockGenerator(first_option_bias=0.8)
    items = _items(CHOICE, 4000)
    reordered = Question(kind="choice", text=CHOICE.text, options=CHOICE.options[::-1])
    for q in (CHOICE, reordered):
        first = q.keys[0]
        mean_first = np.mean([gen.probs("jev", it, q, 0)[first] for it in items])
        true_share = np.mean([it.label == first for it in items])
        assert mean_first > true_share + 0.05  # over-predicted, whichever option it is


def test_reordering_and_rewording_keep_what_the_mock_knows() -> None:
    gen = MockGenerator(wording_jitter=0.5)
    item = Item(id="x", state={"t": "hi"}, label="b")
    reordered = Question(kind="choice", text="Other words?", options=CHOICE.options[::-1])
    a = dict(zip(CHOICE.keys, gen.calibrated_logits("jev", item, CHOICE), strict=True))
    b = dict(zip(reordered.keys, gen.calibrated_logits("jev", item, reordered), strict=True))
    assert a == pytest.approx(b)  # same latent per option key, whatever the order or wording


def test_wording_jitter_changes_the_distortion_per_phrasing() -> None:
    gen = MockGenerator(wording_jitter=0.3)
    factors = {gen.wording_factor(Question(kind="binary", text=t)) for t in ("A?", "B?", "C?")}
    assert len(factors) == 3 and all(0.3 < f < 3 for f in factors)
    assert MockGenerator().wording_factor(BINARY) == 1.0
