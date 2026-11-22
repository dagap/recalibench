"""A binary-only model returning fixed, deliberately over-confident probabilities."""

import math

from recalibench.core.types import Item, Prediction, Question


class ToyModel:
    supports = frozenset({"binary"})
    trainable = False

    def __init__(self, name: str, version: str, sharpness: float = 2.5) -> None:
        self.name = name
        self.version = version
        self.sharpness = sharpness

    def fit(self, items: list[Item], question: Question) -> None:
        return None

    async def predict(self, items: list[Item], question: Question) -> list[Prediction]:
        if question.kind not in self.supports:
            raise ValueError(f"{self.name} does not support {question.kind!r} questions")
        out: list[Prediction] = []
        for item in items:
            p = 1.0 / (1.0 + math.exp(-self.sharpness * float(item.state["signal"])))
            out.append(
                Prediction(
                    item_id=item.id,
                    probs={"yes": p, "no": 1.0 - p},
                    std=None,
                    model=f"{self.name}@{self.version}",
                    latency_ms=0.0,
                    cost_usd=0.0,
                )
            )
        return out

    def estimate_cost(self, items: list[Item], question: Question) -> float:
        return 0.0
