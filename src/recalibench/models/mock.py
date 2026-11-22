"""The `mock` model plug-in: MockGenerator answers through the full ApiModel path.

Offline, free and deliberately miscalibrated (see mock_generator.py), for tests
and for building the pipeline without an API key.
"""

import json
from typing import Any

from recalibench.core.types import Item, Question
from recalibench.models.base_api import ApiModel, ParseError, Usage


class MockModel(ApiModel):
    """The `mock` plug-in: always answers from MockGenerator, in mock and live mode, free."""

    supports = frozenset({"binary", "choice"})
    always_mock = True

    def __init__(self, name: str, version: str = "mock-1", **kwargs: Any) -> None:
        kwargs.setdefault("base_url", "mock://")
        kwargs.setdefault("api_key_env", "")
        super().__init__(name=name, version=version, **kwargs)

    def build_request(self, item: Item, question: Question) -> dict[str, Any]:
        return {"state": item.state, "question": question.text}

    def fake_response(
        self, request: dict[str, Any], question: Question, probs: dict[str, float]
    ) -> dict[str, Any]:
        return {"probs": probs}

    def parse_response(
        self, raw: dict[str, Any], question: Question
    ) -> tuple[dict[str, float], Usage]:
        try:
            probs = {str(k): float(v) for k, v in raw["probs"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ParseError(f"bad mock response: {e}") from e
        return probs, Usage(0, 0)

    def count_tokens(self, request: dict[str, Any]) -> int:
        return len(json.dumps(request)) // 3 + 1

    def price(self, usage: Usage) -> float:
        return 0.0
