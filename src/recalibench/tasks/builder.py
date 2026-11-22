"""Turn a task config plus its dataset into fit/val/test items.

Order of operations:
  sort by id -> build state, drop null/blank -> dedupe identical states ->
  strata from the label rule -> stratified sample -> stratified split ->
  fit the label rule on the fit split only -> label every split -> check -> write.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from recalibench.core.paths import Paths
from recalibench.core.splits import stratified_sample, stratified_split
from recalibench.core.types import Item, TaskSpec
from recalibench.tasks.label_rules import make_rule
from recalibench.tasks.state import StateSpec, build_states

COLUMNS = ["id", "state_json", "label", "split"]
RATE_TOLERANCE = 0.02  # per-split label rates must agree within 2 points


class BuildError(Exception):
    """The built task breaks an invariant (leaky splits, invalid labels, empty split)."""


@dataclass
class BuiltTask:
    frame: pd.DataFrame  # COLUMNS
    rule_params: dict[str, Any]
    dropped: dict[str, int] = field(default_factory=lambda: {})


def build_frame(
    spec: TaskSpec,
    raw: pd.DataFrame,
    *,
    seed: int,
    sample_size: int,
    fractions: Mapping[str, float],
) -> BuiltTask:
    state_spec = StateSpec.parse(spec.state)
    rule = make_rule(spec.label)
    df = raw.sort_values("id", kind="stable").reset_index(drop=True)
    dropped: dict[str, int] = {}

    df = df.assign(state_json=build_states(df, state_spec))
    keep = df["state_json"].notna()
    dropped["empty_state"] = int((~keep).sum())
    df = df[keep]

    dup = df["state_json"].duplicated(keep="first")
    dropped["duplicate_state"] = int(dup.sum())
    df = df[~dup]

    strata = rule.strata(df)
    keep = strata.notna()
    dropped["no_label"] = int((~keep).sum())
    df, strata = df[keep].reset_index(drop=True), strata[keep].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    keys: list[str] = [str(s) for s in strata.tolist()]
    idx = stratified_sample(keys, sample_size, rng)
    df = df.iloc[idx].reset_index(drop=True)
    df["split"] = stratified_split([keys[int(i)] for i in idx], fractions, rng)

    first_split = next(iter(fractions))
    rule.fit(df[df["split"] == first_split])
    labels = rule.apply(df)
    keep = labels.notna()
    dropped["unlabelled_after_fit"] = int((~keep).sum())
    df = df[keep].assign(label=labels[keep])

    if spec.question.kind == "binary":
        df["label"] = df["label"].astype("int64")
    else:
        df["label"] = df["label"].astype("str")
    frame = df[COLUMNS].astype({"id": "str", "state_json": "str", "split": "str"})
    return BuiltTask(frame=frame.reset_index(drop=True), rule_params=rule.params(), dropped=dropped)


def label_rates(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Share of each label within each split."""
    out: dict[str, dict[str, float]] = {}
    for split, g in frame.groupby("split", sort=False):
        shares = g["label"].astype("str").value_counts(normalize=True).sort_index()
        out[str(split)] = {str(k): round(float(v), 6) for k, v in shares.items()}
    return out


def rate_spread(frame: pd.DataFrame) -> dict[str, float]:
    """Per label: max minus min share across splits."""
    rates = label_rates(frame)
    labels = sorted({k for r in rates.values() for k in r})
    return {
        lab: max(r.get(lab, 0.0) for r in rates.values())
        - min(r.get(lab, 0.0) for r in rates.values())
        for lab in labels
    }


def check_splits(frame: pd.DataFrame, spec: TaskSpec, splits: list[str]) -> list[str]:
    """Hard invariants. Label-rate spread is reported separately (it needs enough rows)."""
    problems: list[str] = []
    if not frame["id"].is_unique:
        problems.append("an id appears more than once (possibly in two splits)")
    if not frame["state_json"].is_unique:
        problems.append("an identical state appears more than once")
    allowed: set[Any] = {0, 1} if spec.question.kind == "binary" else set(spec.question.keys)
    bad = set(frame["label"].unique()) - allowed
    if bad:
        problems.append(f"labels outside {sorted(map(str, allowed))}: {sorted(map(str, bad))}")
    for s in splits:
        if not (frame["split"] == s).any():
            problems.append(f"split {s!r} is empty")
    return problems


def write(
    built: BuiltTask, spec: TaskSpec, paths: Paths, fingerprint: str, extra: Mapping[str, Any]
) -> dict[str, Any]:
    out = paths.processed(spec.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    built.frame.to_parquet(out, index=False)
    counts = built.frame["split"].value_counts()
    meta: dict[str, Any] = {
        "task": spec.name,
        "dataset": spec.dataset,
        "kind": spec.question.kind,
        "fingerprint": fingerprint,
        "rows": len(built.frame),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "label_rates": label_rates(built.frame),
        "rule": built.rule_params,
        "dropped": built.dropped,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    paths.processed_meta(spec.name).write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


def read_items(paths: Paths, task: str) -> dict[str, list[Item]]:
    """Items per split, as the scoring stage will consume them."""
    frame = pd.read_parquet(paths.processed(task))
    out: dict[str, list[Item]] = {}
    for rec in frame.to_dict("records"):
        label: Any = rec["label"]
        item = Item(
            id=str(rec["id"]),
            state=json.loads(str(rec["state_json"])),
            label=str(label) if isinstance(label, str) else int(label),
        )
        out.setdefault(str(rec["split"]), []).append(item)
    return out
