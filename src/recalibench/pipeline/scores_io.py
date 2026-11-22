"""scores.parquet + meta.json: the hand-off from scoring to calibration and evaluation.

    artifacts*/<task>/<model>@<version>/scores.parquet
        id, split, label, p_<key>..., std_<key>..., latency_ms, cost_usd
    artifacts*/<task>/<model>@<version>/meta.json
        task, model_id, kind, keys (column order), fingerprint, rows, errors, spend, ...

Probabilities are clipped to [CLIP, 1 - CLIP] here (and only here); choice rows
are renormalised afterwards so they still sum to 1. Downstream stages read
scores only through `read_scores`.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from recalibench.core.protocols import FloatArray
from recalibench.core.types import Item, Prediction, Question

CLIP = 1e-6
SCORES = "scores.parquet"
META = "meta.json"
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_]+$")


def clip_probs(values: Sequence[float], kind: str) -> list[float]:
    p = np.clip(np.asarray(values, dtype=np.float64), CLIP, 1 - CLIP)
    if kind == "binary":
        return [float(p[0]), float(1 - p[0])]
    return [float(x) for x in p / p.sum()]


def build_frame(
    items: Sequence[Item], splits: Sequence[str], preds: dict[str, Prediction], question: Question
) -> pd.DataFrame:
    """One row per item that has a prediction, in item order."""
    keys = list(question.keys)
    for k in keys:
        if not _SAFE_KEY.match(k):
            raise ValueError(f"option key {k!r} can't be a column name ([A-Za-z0-9_]+)")
    rows: list[dict[str, Any]] = []
    for item, split in zip(items, splits, strict=True):
        pred = preds.get(item.id)
        if pred is None:
            continue
        probs = clip_probs([pred.probs[k] for k in keys], question.kind)
        row: dict[str, Any] = {"id": item.id, "split": split, "label": item.label}
        row.update({f"p_{k}": v for k, v in zip(keys, probs, strict=True)})
        for k in keys:
            row[f"std_{k}"] = None if pred.std is None else float(pred.std[k])
        row["latency_ms"] = pred.latency_ms
        row["cost_usd"] = pred.cost_usd
        rows.append(row)
    frame = pd.DataFrame(rows)
    std_cols = [f"std_{k}" for k in keys]
    frame[std_cols] = frame[std_cols].astype("float64")
    frame["label"] = frame["label"].astype("int64" if question.kind == "binary" else "str")
    return frame


def write_scores(directory: Path, frame: pd.DataFrame, meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{SCORES}.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.replace(directory / SCORES)
    (directory / META).write_text(json.dumps(meta, indent=2, default=str) + "\n")


def read_meta(directory: Path) -> dict[str, Any] | None:
    path = directory / META
    if not path.exists() or not (directory / SCORES).exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text()))


@dataclass(frozen=True)
class Scores:
    frame: pd.DataFrame
    meta: dict[str, Any]

    @property
    def keys(self) -> list[str]:
        return [str(k) for k in self.meta["keys"]]

    def probs(self, split: str | None = None) -> FloatArray:
        """(n, n_keys) probability matrix, columns in `keys` order."""
        df = self.frame if split is None else self.frame[self.frame["split"] == split]
        return df[[f"p_{k}" for k in self.keys]].to_numpy(dtype=np.float64)

    def labels(self, split: str | None = None) -> pd.Series:
        df = self.frame if split is None else self.frame[self.frame["split"] == split]
        return df["label"]


def read_scores(directory: Path) -> Scores:
    meta = read_meta(directory)
    if meta is None:
        raise FileNotFoundError(f"no scores in {directory}")
    return Scores(frame=pd.read_parquet(directory / SCORES), meta=meta)
