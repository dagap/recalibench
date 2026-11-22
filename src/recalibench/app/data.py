"""What the Streamlit app shows, as plain functions (the UI file stays thin).

Reads artifacts*/ (metrics.json via report.collect, scores.parquet, calibrated.parquet,
selected.json) and the case text in data/processed/. Nothing knows a task or model by name.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from recalibench.core.paths import Mode, Paths
from recalibench.core.protocols import FloatArray, IntArray
from recalibench.pipeline.calibrate import label_indices, read_selected
from recalibench.pipeline.scores_io import read_scores
from recalibench.report.collect import Pair, ReportData, collect

CALIBRATED = "calibrated.parquet"
MODES: tuple[Mode, ...] = ("live", "mock")


class AppDataError(Exception):
    """The artifacts can't be shown as they are; the message says how to fix them."""


@dataclass(frozen=True)
class PolicyPoint:
    threshold: float
    n: int
    auto_count: int
    review_count: int
    auto_share: float
    auto_error_rate: float | None  # None when nothing is auto-handled


def available_modes(root: Path) -> list[Mode]:
    """Modes with at least one metrics.json on disk, live first."""
    return [m for m in MODES if any(Paths(root, m).artifacts_root().glob("*/*/metrics.json"))]


def default_mode(root: Path) -> Mode:
    modes = available_modes(root)
    return modes[0] if modes else "mock"


def load(paths: Paths) -> ReportData:
    return collect(paths)


def pair_dir(paths: Paths, pair: Pair) -> Path:
    return paths.artifacts_root() / pair.task / pair.model_id


def _case_text(paths: Paths, task: str) -> dict[str, str]:
    path = paths.processed(task)
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["id", "state_json"])
    out: dict[str, str] = {}
    for rec in frame.to_dict("records"):
        state = cast(dict[str, Any], json.loads(str(rec["state_json"])))
        if len(state) == 1:
            out[str(rec["id"])] = str(next(iter(state.values())))
        else:
            out[str(rec["id"])] = "\n".join(f"{k}: {v}" for k, v in state.items())
    return out


def test_rows(paths: Paths, pair: Pair) -> pd.DataFrame:
    """One row per test item: label, raw and calibrated probabilities, decisions, case text."""
    directory = pair_dir(paths, pair)
    scores = read_scores(directory)
    selected = read_selected(directory)
    if selected is None or selected.get("scores_fingerprint") != scores.meta["fingerprint"]:
        raise AppDataError(
            f"{pair.task} x {pair.model_id}: the calibrator was fitted on different scores; "
            "run `recal calibrate` and `recal evaluate`"
        )
    keys = scores.keys
    test = (scores.frame["split"] == "test").to_numpy()
    frame = scores.frame[test].reset_index(drop=True)
    y: IntArray = label_indices(scores)[test]
    raw = scores.probs("test")

    every = pd.read_parquet(directory / CALIBRATED)
    mask = ((every["calibrator"] == selected["selected"]) & (every["split"] == "test")).to_numpy()
    by_id = cast(pd.DataFrame, every.loc[mask]).set_index("id")
    columns = [f"p_{k}" for k in keys]
    cal: FloatArray = by_id.reindex(frame["id"])[columns].to_numpy(dtype=np.float64)
    if np.isnan(cal).any():
        raise AppDataError(
            f"{pair.task} x {pair.model_id}: calibrated scores are missing test rows; "
            "run `recal calibrate --force`"
        )

    text = _case_text(paths, pair.task)
    out = pd.DataFrame({"id": frame["id"].astype(str), "label": [keys[int(i)] for i in y]})
    sides: list[tuple[str, FloatArray]] = [("raw", raw), ("cal", cal)]
    for side, P in sides:
        for j, k in enumerate(keys):
            out[f"{side}_p_{k}"] = P[:, j]
        out[f"{side}_choice"] = [keys[i] for i in P.argmax(axis=1)]
        out[f"{side}_conf"] = P.max(axis=1)
        out[f"{side}_correct"] = out[f"{side}_choice"] == out["label"]
    out["text"] = [text.get(i, "") for i in out["id"]]
    return out


def policy(
    conf: np.ndarray[Any, Any], correct: np.ndarray[Any, Any], threshold: float
) -> PolicyPoint:
    """Auto-handle every case with confidence >= threshold; people review the rest."""
    n = int(conf.size)
    auto = conf >= threshold
    k = int(auto.sum())
    errors = int((~correct[auto]).sum())
    return PolicyPoint(
        threshold=float(threshold),
        n=n,
        auto_count=k,
        review_count=n - k,
        auto_share=k / n if n else 0.0,
        auto_error_rate=errors / k if k else None,
    )


def policy_curve(
    conf: np.ndarray[Any, Any], correct: np.ndarray[Any, Any], thresholds: Sequence[float]
) -> list[PolicyPoint]:
    return [policy(conf, correct, t) for t in thresholds]
