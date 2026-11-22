"""Local trainable baselines: logistic regression on the item state.

    features: text        TF-IDF (1-2 grams, min_df 3, sublinear tf) over the text fields
    features: structured  the other fields: numbers standardised, categoricals one-hot

Field roles are detected from the fit items: a string field is *text* when its
distinct-value ratio is above TEXT_DISTINCT or its mean word count is at least
TEXT_WORDS, other strings are *categorical*, numbers are *numeric*. A task with
no field of the needed kind raises UnsupportedTask (the runner logs a skip).

scikit-learn has no type stubs; its objects are held as `Any` in this file.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # pyright: ignore[reportMissingTypeStubs]
from sklearn.feature_extraction.text import (  # pyright: ignore[reportMissingTypeStubs]
    TfidfVectorizer,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.linear_model import (  # pyright: ignore[reportMissingTypeStubs]
    LogisticRegression,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.pipeline import (  # pyright: ignore[reportMissingTypeStubs]
    make_pipeline,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.preprocessing import (  # pyright: ignore[reportMissingTypeStubs]
    OneHotEncoder,  # pyright: ignore[reportUnknownVariableType]
    StandardScaler,  # pyright: ignore[reportUnknownVariableType]
)

from recalibench.core.protocols import UnsupportedTask
from recalibench.core.types import Item, Prediction, Question

TEXT_DISTINCT = 0.2
TEXT_WORDS = 5


def field_roles(items: Sequence[Item]) -> dict[str, str]:
    """`text`, `categorical` or `numeric` per state field, from the fit items."""
    fields = list(items[0].state) if items else []
    roles: dict[str, str] = {}
    for f in fields:
        values = [it.state.get(f) for it in items]
        if all(isinstance(v, bool | int | float) for v in values):
            roles[f] = "numeric"
            continue
        texts = [str(v) for v in values]
        distinct = len(set(texts)) / len(texts)
        words = sum(len(t.split()) for t in texts) / len(texts)
        roles[f] = "text" if distinct > TEXT_DISTINCT or words >= TEXT_WORDS else "categorical"
    return roles


class TfidfLogReg:
    supports = frozenset({"binary", "choice"})
    trainable = True

    def __init__(
        self,
        name: str,
        version: str,
        features: str = "text",
        min_df: int = 3,
        C: float = 1.0,
        max_iter: int = 2000,
    ) -> None:
        if features not in ("text", "structured"):
            raise ValueError("features must be 'text' or 'structured'")
        self.name = name
        self.version = version
        self.features = features
        self.min_df = min_df
        self.C = C
        self.max_iter = max_iter
        self._pipe: Any = None
        self._fields: list[str] = []
        self._classes: list[str] = []
        self._constant: dict[str, float] | None = None

    # -- features ------------------------------------------------------------------
    def _select(self, roles: dict[str, str]) -> list[str]:
        if self.features == "text":
            fields = [f for f, r in roles.items() if r == "text"]
        else:
            fields = [f for f, r in roles.items() if r != "text"]
        if not fields:
            kind = "text" if self.features == "text" else "non-text"
            raise UnsupportedTask(f"{self.name}: the task's state has no {kind} fields")
        return fields

    def _frame(self, items: Sequence[Item]) -> pd.DataFrame:
        return pd.DataFrame([{f: it.state.get(f) for f in self._fields} for it in items])

    def _text(self, items: Sequence[Item]) -> list[str]:
        return [" \n ".join(str(it.state.get(f, "")) for f in self._fields) for it in items]

    def _build(self, roles: dict[str, str]) -> Any:
        clf: Any = LogisticRegression(C=self.C, max_iter=self.max_iter)
        if self.features == "text":
            vec: Any = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, sublinear_tf=True)
            return make_pipeline(vec, clf)
        numeric = [f for f in self._fields if roles[f] == "numeric"]
        categorical = [f for f in self._fields if roles[f] == "categorical"]
        parts: list[tuple[str, Any, list[str]]] = []
        if numeric:
            parts.append(("num", StandardScaler(), numeric))
        if categorical:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
        return make_pipeline(ColumnTransformer(parts), clf)

    def _x(self, items: Sequence[Item]) -> Any:
        return self._text(items) if self.features == "text" else self._frame(items)

    # -- Model protocol ------------------------------------------------------------
    def fit(self, items: list[Item], question: Question) -> None:
        if not items:
            raise ValueError(f"{self.name}: no items to fit on")
        roles = field_roles(items)
        self._fields = self._select(roles)
        y = [str(it.label) for it in items]
        self._classes = sorted(set(y))
        self._constant = None
        if len(self._classes) < 2:  # e.g. a fold with one class: predict the base rate
            self._constant = {c: 1.0 for c in self._classes}
            self._pipe = None
            return
        self._pipe = self._build(roles)
        self._pipe.fit(self._x(items), y)
        self._classes = [str(c) for c in self._pipe.classes_]

    def _class_to_key(self, question: Question) -> dict[str, str]:
        return (
            {"1": "yes", "0": "no"} if question.kind == "binary" else {k: k for k in question.keys}
        )

    async def predict(self, items: list[Item], question: Question) -> list[Prediction]:
        if not self._classes:
            raise RuntimeError(f"{self.name}: predict() before fit()")
        if self._pipe is None:
            matrix = np.ones((len(items), 1))
        else:
            matrix = np.asarray(self._pipe.predict_proba(self._x(items)), dtype=np.float64)
        to_key = self._class_to_key(question)
        out: list[Prediction] = []
        for item, row in zip(items, matrix, strict=True):
            probs = dict.fromkeys(question.keys, 0.0)  # classes unseen in training get 0
            for cls, p in zip(self._classes, row, strict=True):
                probs[to_key[cls]] = float(p)
            out.append(
                Prediction(
                    item_id=item.id,
                    probs=probs,
                    std=None,
                    model=f"{self.name}@{self.version}",
                    latency_ms=None,
                    cost_usd=0.0,
                )
            )
        return out

    def estimate_cost(self, items: list[Item], question: Question) -> float:
        return 0.0


def stratified_folds(labels: Sequence[int | str], k: int, seed: int) -> list[int]:
    """A fold index per row, each label spread evenly over the k folds."""
    rng = np.random.default_rng(seed)
    folds = [0] * len(labels)
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(str(lab), []).append(i)
    offset = 0
    for key in sorted(by_label):
        idx = rng.permutation(np.asarray(by_label[key]))
        for j, i in enumerate(idx):
            folds[int(i)] = (j + offset) % k
        offset += len(idx)
    if k > len(labels):
        raise ValueError(f"cannot make {k} folds from {len(labels)} rows")
    return folds


def n_folds_for(n: int, k: int) -> int:
    return max(2, min(k, math.floor(n / 2)))
