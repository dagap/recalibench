"""Offline, deterministic, deliberately miscalibrated answers.

MockGenerator first builds a *calibrated* score from the item's label (the one
place a "model" reads a label: a score unrelated to the truth would leave
nothing to calibrate), then distorts it by a known amount:

  binary  z ~ N(+mu, 1) if y=1 else N(-mu, 1); calibrated logit = 2*mu*z
          (exact Bayes posterior for balanced classes); returned p = sigmoid(slope * logit)
          slope < 1 -> under-confident (default config: 0.6)
  choice  s_k = mu*[k == y] + N(0, 1); calibrated probs = softmax(mu * s) (uniform prior);
          returned probs = softmax(mu * s / temperature); temperature < 1 -> over-confident (0.5)

The latent draw depends only on (model, item, question); each repeat adds small
logit noise (noise_sd), so repeats differ slightly and std is meaningful.

Every ApiModel routes here when mode=mock (see base_api.py); models/mock.py is the
stand-alone `mock` plug-in that never leaves this path.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from recalibench.core.types import Item, Question
from recalibench.models.cache import canonical_state


def _rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _softmax(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass(frozen=True)
class MockGenerator:
    signal: float = 1.0  # mu: separation of the classes (higher = more accurate)
    binary_slope: float = 0.6
    choice_temperature: float = 0.5
    noise_sd: float = 0.1
    # Simulations of documented Jev behaviour, so the experiments show an effect offline:
    wording_jitter: float = 0.0  # sd of a log-scale change in slope/temperature per question text
    first_option_bias: float = 0.0  # logit bonus for whichever option is listed first

    def namespace(self) -> str:
        return (
            f"mock:{self.signal}:{self.binary_slope}:{self.choice_temperature}:{self.noise_sd}"
            f":{self.wording_jitter}:{self.first_option_bias}"
        )

    def wording_factor(self, question: Question) -> float:
        """Deterministic per-wording scale on the distortion (1.0 without jitter)."""
        if self.wording_jitter <= 0:
            return 1.0
        z = float(_rng("wording", question.kind, question.text).normal())
        return float(np.exp(self.wording_jitter * z))

    def calibrated_logits(
        self, model: str, item: Item, question: Question
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """The calibrated score before any distortion (binary: one logit).

        It depends on the item and the option keys only, not on the wording or the
        option order: rephrasing or reordering changes how miscalibrated the mock
        is, not what it "knows".
        """
        state = canonical_state(item.state)
        mu = self.signal
        if question.kind == "binary":
            y = int(item.label)
            z = _rng(model, state, "binary").normal(mu if y == 1 else -mu, 1.0)
            return np.array([2.0 * mu * z])
        keys = question.keys
        if str(item.label) not in keys:
            raise ValueError(f"label {item.label!r} is not an option key of the question")
        s = np.array([_rng(model, state, "choice", k).normal() for k in keys])
        s += mu * np.array([k == str(item.label) for k in keys], dtype=np.float64)
        return mu * s

    def probs(
        self, model: str, item: Item, question: Question, repeat_idx: int
    ) -> dict[str, float]:
        logits = self.calibrated_logits(model, item, question)
        noise_rng = _rng(model, canonical_state(item.state), question.text, f"repeat={repeat_idx}")
        noise = noise_rng.normal(0.0, self.noise_sd, logits.shape) if self.noise_sd > 0 else 0.0
        factor = self.wording_factor(question)
        if question.kind == "binary":
            slope = self.binary_slope * factor
            p = _sigmoid(slope * float(logits[0]) + float(np.asarray(noise).ravel()[0]))
            return {"yes": p, "no": 1.0 - p}
        z = logits / (self.choice_temperature * factor) + noise
        z[0] += self.first_option_bias  # position 0 = the option listed first
        dist = _softmax(z)
        return {k: float(v) for k, v in zip(question.keys, dist, strict=True)}
