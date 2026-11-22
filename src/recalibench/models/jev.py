"""Jev (TypeSafe System One). The only file that knows TypeSafe's wire format.

Source: https://docs.typesafe.ai (api/system-one, api/errors, models, primitives/noul,
primitives/choice), checked 2026-11-22.

    POST {base_url}/systemone        Authorization: Bearer <key>
    {"model": "jev-1.13.0", "state": <object>, "questions": {"q": <question>}}
      binary -> {"type": "noul",   "instructions": <text>}
      choice -> {"type": "choice", "instructions": <text>, "criteria": {<name>: <description>}}
    reply: {"model": ..., "answers": {"q": {...}}, "usage": {"input_tokens", "output_tokens"}}
      noul   -> {"type": "noul", "noul": p_yes}
      choice -> {"type": "choice", "choice": name, "confidence": c, "probabilities": {name: p}}

Choice criteria are keyed by option *label* (what the model reads) and mapped
back to option keys. `confidence` is ignored: the docs' own example disagrees
with its probabilities. Only input tokens are billed. 429 and 529 are retried by
ApiModel; other 4xx are fatal.
"""

import json
import os
from typing import Any, cast

from recalibench.core.types import Item, Question
from recalibench.models.base_api import ApiModel, FatalApiError, ParseError, Usage

QUESTION_NAME = "q"
CHARS_PER_TOKEN = 3  # conservative: TypeSafe's tokenizer is not documented


class JevModel(ApiModel):
    supports = frozenset({"binary", "choice"})

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/systemone"

    def headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise FatalApiError(
                f"{self.model_id}: set {self.api_key_env} (e.g. in .env) for live runs"
            )
        return {"Authorization": f"Bearer {key}"}

    def _criteria(self, question: Question) -> dict[str, str | None]:
        labels = [o.label for o in question.options]
        if len(set(labels)) != len(labels):
            raise ValueError("Jev choice options need unique labels (they are the criteria keys)")
        return {o.label: (o.description or None) for o in question.options}

    def build_request(self, item: Item, question: Question) -> dict[str, Any]:
        if question.kind == "binary":
            q: dict[str, Any] = {"type": "noul", "instructions": question.text}
        else:
            q = {
                "type": "choice",
                "instructions": question.text,
                "criteria": self._criteria(question),
            }
        return {"model": self.version, "state": item.state, "questions": {QUESTION_NAME: q}}

    def parse_response(
        self, raw: dict[str, Any], question: Question
    ) -> tuple[dict[str, float], Usage]:
        if raw.get("model") != self.version:
            raise ParseError(f"answered by {raw.get('model')!r}, pinned {self.version!r}")
        try:
            answer = cast(dict[str, Any], raw["answers"][QUESTION_NAME])
            usage_raw = cast(dict[str, Any], raw.get("usage") or {})
            usage = Usage(
                input_tokens=int(usage_raw["input_tokens"]),
                output_tokens=int(usage_raw.get("output_tokens", 0)),
            )
            if question.kind == "binary":
                p = float(answer["noul"])
                return {"yes": p, "no": 1.0 - p}, usage
            by_label = cast(dict[str, Any], answer["probabilities"])
            label_to_key = {o.label: o.key for o in question.options}
            unknown = set(by_label) - set(label_to_key)
            if unknown:
                raise ParseError(f"unknown options in reply: {sorted(unknown)}")
            return {label_to_key[k]: float(v) for k, v in by_label.items()}, usage
        except (KeyError, TypeError, ValueError) as e:
            raise ParseError(f"{type(e).__name__}: {e}") from e

    def count_tokens(self, request: dict[str, Any]) -> int:
        chars = len(json.dumps(request["state"], ensure_ascii=False)) + len(
            json.dumps(request["questions"], ensure_ascii=False)
        )
        return chars // CHARS_PER_TOKEN + 1

    def fake_response(
        self, request: dict[str, Any], question: Question, probs: dict[str, float]
    ) -> dict[str, Any]:
        usage = {"input_tokens": self.count_tokens(request), "output_tokens": 0}
        if question.kind == "binary":
            answer: dict[str, Any] = {"type": "noul", "noul": probs["yes"]}
        else:
            by_label = {o.label: probs[o.key] for o in question.options}
            top = max(by_label, key=lambda k: by_label[k])
            answer = {
                "type": "choice",
                "choice": top,
                "confidence": by_label[top],
                "probabilities": by_label,
            }
        return {"model": request["model"], "answers": {QUESTION_NAME: answer}, "usage": usage}
