"""Score stage: every experiment task x model -> scores.parquet.

    plan      unsupported pairs are skipped with a reason; fresh pairs are skipped
    estimate  cost per model before anything is spent (live: only uncached calls);
              dry_run stops here; live runs must fit inside budget_usd
    score     trainable models: fit-split scores are out-of-fold (k-fold cross-fitting),
              val/test come from a model fitted on the whole fit split
    check     items missing from the predictions are errors; above max_error_rate the
              pair fails and writes nothing (the cache keeps what succeeded)
    write     scores.parquet + meta.json (pipeline/scores_io.py)

`limit` (smoke runs) scores the first items of each split and writes no artifact.
"""

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from omegaconf import DictConfig, OmegaConf

from recalibench.core.paths import Paths
from recalibench.core.protocols import Model, UnsupportedTask
from recalibench.core.registry import Registry
from recalibench.core.types import Item, Prediction, Question, TaskSpec
from recalibench.models.budget import Budget
from recalibench.models.context import NeedsContext, RunContext
from recalibench.models.tfidf_logreg import n_folds_for, stratified_folds
from recalibench.pipeline import scores_io
from recalibench.tasks.builder import read_items

log = logging.getLogger(__name__)

SCORE_VERSION = 1  # bump when scoring output changes for the same inputs
SPLIT_ORDER = ("fit", "val", "test")
OOF_FOLDS = 5


class ScoreError(Exception):
    """The score stage can't run (missing inputs) or a pair failed its error check."""


@runtime_checkable
class HasPendingCost(Protocol):
    def pending_cost(self, items: list[Item], questions: Sequence[Question]) -> float: ...


@dataclass
class PairResult:
    task: str
    model: str
    status: str  # scored | fresh | skipped | failed | estimated | smoke
    reason: str = ""
    rows: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    estimate_usd: float = 0.0


@dataclass
class ScorePlan:
    """One task (or task variant) x model to score into `out_dir`."""

    spec: TaskSpec
    model_name: str
    model: Model
    items: list[Item]
    splits: list[str]
    fingerprint: str
    task_fingerprint: str
    out_dir: Path
    estimate_usd: float = 0.0


def load_task(paths: Paths, name: str, limit: int | None) -> tuple[list[Item], list[str], str]:
    if not paths.processed(name).exists():
        raise ScoreError(f"{name}: not prepared; run `recal prepare` first")
    meta = cast(dict[str, Any], json.loads(paths.processed_meta(name).read_text()))
    by_split = read_items(paths, name)
    items: list[Item] = []
    splits: list[str] = []
    total = sum(len(v) for v in by_split.values())
    for split in [s for s in SPLIT_ORDER if s in by_split] + sorted(
        set(by_split) - set(SPLIT_ORDER)
    ):
        chosen = by_split[split]
        if limit is not None:
            chosen = chosen[: max(1, math.ceil(limit * len(chosen) / total))]
        items.extend(chosen)
        splits.extend([split] * len(chosen))
    return items, splits, str(meta["fingerprint"])


def _fingerprint(task_fp: str, model_cfg: Any, question: Question) -> str:
    q = [question.kind, question.text, [[o.key, o.label, o.description] for o in question.options]]
    blob = json.dumps(
        {"score": SCORE_VERSION, "task": task_fp, "model": model_cfg, "question": q},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _fresh_model(reg: Registry, name: str, ctx: RunContext) -> Model:
    model = reg.model(name)
    if isinstance(model, NeedsContext):
        model.bind(ctx)
    return model


def _predict(model: Model, items: list[Item], question: Question) -> dict[str, Prediction]:
    if not items:
        return {}
    preds = asyncio.run(model.predict(items, question))
    return {p.item_id: p for p in preds}


def _score_trainable(
    reg: Registry, plan: ScorePlan, ctx: RunContext, seed: int
) -> dict[str, Prediction]:
    q = plan.spec.question
    fit = [it for it, s in zip(plan.items, plan.splits, strict=True) if s == "fit"]
    rest = [it for it, s in zip(plan.items, plan.splits, strict=True) if s != "fit"]
    preds: dict[str, Prediction] = {}
    k = n_folds_for(len(fit), OOF_FOLDS)
    folds = stratified_folds([it.label for it in fit], k, seed)
    for f in range(k):
        fold_model = _fresh_model(reg, plan.model_name, ctx)
        fold_model.fit([it for it, g in zip(fit, folds, strict=True) if g != f], q)
        preds.update(
            _predict(fold_model, [it for it, g in zip(fit, folds, strict=True) if g == f], q)
        )
    plan.model.fit(fit, q)
    preds.update(_predict(plan.model, rest, q))
    return preds


def _model_label(model: Model) -> str:
    return f"{model.name}@{model.version}"


def plan_pair(
    reg: Registry,
    ctx: RunContext,
    cfg: DictConfig,
    spec: TaskSpec,
    model_name: str,
    items: list[Item],
    splits: list[str],
    task_fp: str,
    out_dir: Path | None = None,
    force: bool = False,
) -> ScorePlan | PairResult:
    """A plan to score `spec` with `model_name`, or a PairResult if it is skipped or fresh."""
    limit = None if cfg.get("limit") is None else int(cfg.limit)
    model = reg.model(model_name)
    label = _model_label(model)
    if spec.question.kind not in model.supports:
        reason = f"{label} supports {sorted(model.supports)}, task is {spec.question.kind}"
        log.info("skip %s x %s: %s", spec.name, label, reason)
        return PairResult(spec.name, label, "skipped", reason)
    if isinstance(model, NeedsContext):
        model.bind(ctx)
    model_cfg = OmegaConf.to_container(reg.config("model", model_name), resolve=True)
    fp = _fingerprint(task_fp, model_cfg, spec.question)
    out = out_dir or ctx.paths.artifact_dir(spec.name, model.name, model.version)
    old = scores_io.read_meta(out)
    fresh = old is not None and old.get("fingerprint") == fp
    if fresh and old is not None and limit is None and not force and not cfg.get("dry_run"):
        log.info("%s x %s: up to date, skipped", spec.name, label)
        return PairResult(spec.name, label, "fresh", rows=int(old["rows"]))
    return ScorePlan(spec, model_name, model, items, splits, fp, task_fp, out)


def estimate_and_run(
    reg: Registry, plans: list[ScorePlan], ctx: RunContext, cfg: DictConfig
) -> list[PairResult]:
    """Estimate every plan before spending anything, then score them one by one."""
    live = ctx.mode == "live"
    limit = None if cfg.get("limit") is None else int(cfg.limit)
    per_model: dict[str, float] = {}
    for plan in plans:
        q = plan.spec.question
        if live and isinstance(plan.model, HasPendingCost):
            plan.estimate_usd = plan.model.pending_cost(plan.items, [q])
        else:  # mock runs show what a full live run would cost
            plan.estimate_usd = plan.model.estimate_cost(plan.items, q)
        label = _model_label(plan.model)
        per_model[label] = per_model.get(label, 0.0) + plan.estimate_usd
    total = sum(per_model.values())
    for label, usd in per_model.items():
        log.info("estimate %s: $%.4f%s", label, usd, "" if live else " (if run live)")
    if cfg.get("dry_run"):
        log.info(
            "estimate total: $%.4f (budget $%.2f); dry run, nothing scored",
            total,
            ctx.budget.cap_usd,
        )
        return [
            PairResult(p.spec.name, _model_label(p.model), "estimated", estimate_usd=p.estimate_usd)
            for p in plans
        ]
    if live:
        ctx.budget.check_estimate(total)
    results = [_run_pair(reg, plan, ctx, cfg, limit) for plan in plans]
    for label, usd in sorted(ctx.budget.spent_by_model.items()):
        log.info("spent %s: $%.4f (estimate $%.4f)", label, usd, per_model.get(label, 0.0))
    return results


def score(reg: Registry, paths: Paths, cfg: DictConfig, force: bool = False) -> list[PairResult]:
    limit = None if cfg.get("limit") is None else int(cfg.limit)
    ctx = RunContext(paths=paths, budget=Budget(cap_usd=float(cfg.budget_usd)))
    results: list[PairResult] = []
    plans: list[ScorePlan] = []
    for task in [str(t) for t in cfg.experiment.tasks]:
        spec = reg.task(task)
        items, splits, task_fp = load_task(paths, task, limit)
        for name in [str(m) for m in cfg.experiment.models]:
            planned = plan_pair(reg, ctx, cfg, spec, name, items, splits, task_fp, force=force)
            if isinstance(planned, PairResult):
                results.append(planned)
            else:
                plans.append(planned)
    return results + estimate_and_run(reg, plans, ctx, cfg)


def _run_pair(
    reg: Registry, plan: ScorePlan, ctx: RunContext, cfg: DictConfig, limit: int | None
) -> PairResult:
    task, label, q = plan.spec.name, _model_label(plan.model), plan.spec.question
    spent_before = ctx.budget.spent_by_model.get(label, 0.0)
    try:
        if plan.model.trainable:
            preds = _score_trainable(reg, plan, ctx, int(cfg.seed))
        else:
            preds = _predict(plan.model, plan.items, q)
    except UnsupportedTask as e:
        log.info("skip %s x %s: %s", task, label, e)
        return PairResult(task, label, "skipped", str(e))
    spent = ctx.budget.spent_by_model.get(label, 0.0) - spent_before
    missing = [it.id for it in plan.items if it.id not in preds]
    rate = len(missing) / max(1, len(plan.items))
    reasons = cast(dict[str, str], getattr(plan.model, "last_errors", {}))
    samples = {i: reasons.get(i, "no prediction returned") for i in missing[:20]}
    frame = scores_io.build_frame(plan.items, plan.splits, preds, q)
    result = PairResult(task, label, "scored", rows=len(frame), errors=len(missing), cost_usd=spent)
    result.estimate_usd = plan.estimate_usd
    if rate > float(cfg.max_error_rate):
        result.status = "failed"
        result.reason = (
            f"{len(missing)}/{len(plan.items)} items failed ({rate:.1%} > "
            f"{float(cfg.max_error_rate):.0%}); e.g. {next(iter(samples.values()), '')}"
        )
        log.error("%s x %s: %s; nothing written", task, label, result.reason)
        return result
    if limit is not None:
        result.status = "smoke"
        sample = frame.head(3)[["id"] + [f"p_{k}" for k in q.keys]].to_dict("records")
        log.info(
            "smoke %s x %s: %d items, %d errors, $%.4f; e.g. %s",
            task,
            label,
            len(plan.items),
            len(missing),
            spent,
            sample,
        )
        return result
    latency = frame["latency_ms"].dropna()
    meta: dict[str, Any] = {
        "task": task,
        "dataset": plan.spec.dataset,
        "model": plan.model.name,
        "model_id": label,
        "version": plan.model.version,
        "kind": q.kind,
        "keys": list(q.keys),
        "mode": ctx.mode,
        "fingerprint": plan.fingerprint,
        "task_fingerprint": plan.task_fingerprint,
        "rows": len(frame),
        "rows_by_split": {str(k): int(v) for k, v in frame["split"].value_counts().items()},
        "errors": len(missing),
        "error_rate": round(rate, 6),
        "error_samples": samples,
        "trainable": plan.model.trainable,
        "out_of_fold": plan.model.trainable,
        "cost_usd": float(frame["cost_usd"].sum()),
        "spent_this_run_usd": spent,
        "estimate_usd": plan.estimate_usd,
        "mean_latency_ms": None if latency.empty else float(latency.mean()),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = plan.out_dir
    scores_io.write_scores(out, frame, meta)
    log.info(
        "%s x %s: %d rows, %d errors, $%.4f -> %s",
        task,
        label,
        len(frame),
        len(missing),
        spent,
        out,
    )
    return result
