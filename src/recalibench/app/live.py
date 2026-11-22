"""The app's live box: score one typed case with the real hosted model, then apply the
calibrator the live run selected for that task.

Same adapter, pinned version, cache and budget guard as the pipeline: answers are cached
under cache/ (asking again is free) and a per-session Budget caps the spend.
"""

import asyncio
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.core.types import Item
from recalibench.models.budget import Budget
from recalibench.models.context import NeedsContext, RunContext
from recalibench.pipeline.calibrate import read_selected

LIVE_MODEL = "jev"
LIVE_BOX_CAP_USD = 0.05  # per app session; one case costs well under a cent


class LiveBoxError(Exception):
    """The live box can't score this case; the message says why."""


@dataclass(frozen=True)
class LiveResult:
    keys: list[str]
    raw: dict[str, float]
    calibrated: dict[str, float]
    selected: str
    model_id: str
    cost_usd: float


def state_fields(task: str, reg: Registry | None = None) -> list[str]:
    """The fields the model sees for `task` (the state's aliases), in config order."""
    fields: Any = (reg or Registry()).task(task).state["fields"]
    if isinstance(fields, dict):
        return [str(k) for k in cast(dict[str, Any], fields)]
    return [str(f) for f in cast(list[Any], fields)]


def _model_dir(root: Path, task: str, reg: Registry) -> tuple[Path, str, str]:
    model = reg.model(LIVE_MODEL)
    paths = Paths(root, "live")
    return (
        paths.artifact_dir(task, model.name, model.version),
        f"{model.name}@{model.version}",
        str(getattr(model, "api_key_env", "")),
    )


def unavailable(root: Path, task: str, reg: Registry | None = None) -> str | None:
    """Why the live box can't run for `task`, or None when it can."""
    directory, model_id, key_env = _model_dir(root, task, reg or Registry())
    if key_env and not os.environ.get(key_env):
        return f"Set {key_env} (in .env, then `make demo`) to try a case live."
    if read_selected(directory) is None:
        return f"No live calibrator for {task} × {model_id} yet: do the live run first."
    return None


def score_live(
    root: Path, task: str, state: dict[str, str], budget: Budget, reg: Registry | None = None
) -> LiveResult:
    reg = reg or Registry()
    spec = reg.task(task)
    directory, model_id, _ = _model_dir(root, task, reg)
    selected = read_selected(directory)
    if selected is None:
        raise LiveBoxError(f"no live calibrator for {task} × {model_id}; do the live run first")
    if selected["model_id"] != model_id:
        raise LiveBoxError(
            f"the calibrator was fitted on {selected['model_id']}, but the live model is "
            f"{model_id}; re-run the live pipeline"
        )
    model = reg.model(LIVE_MODEL)
    if isinstance(model, NeedsContext):
        model.bind(RunContext(paths=Paths(root, "live"), budget=budget))
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]
    # The label is required by Item but never sent: adapters send item.state only.
    item = Item(id=f"live-{digest}", state=dict(state), label=spec.question.keys[0])
    spent_before = budget.spent_usd
    preds = asyncio.run(model.predict([item], spec.question))
    if not preds:
        errors = cast(dict[str, str], getattr(model, "last_errors", {}))
        raise LiveBoxError(errors.get(item.id, "the model returned no answer"))

    keys = [str(k) for k in selected["keys"]]
    raw = {k: float(preds[0].probs[k]) for k in keys}
    with (directory / "calibrators" / f"{selected['selected']}.pkl").open("rb") as f:
        calibrator: Any = pickle.load(f)  # our own artifact, written by the calibrate stage
    Q = np.asarray(calibrator.predict(np.array([[raw[k] for k in keys]])), dtype=np.float64)
    return LiveResult(
        keys=keys,
        raw=raw,
        calibrated={k: float(Q[0, j]) for j, k in enumerate(keys)},
        selected=str(selected["selected"]),
        model_id=model_id,
        cost_usd=budget.spent_usd - spent_before,
    )
