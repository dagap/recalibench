"""Experiments stage (`experiment.extras`), written under artifacts*/_experiments/<name>/.

    wording       one binary-or-choice task x model, several phrasings of the question
                  (the task config's `phrasings:`); each variant is scored, calibrated and
                  evaluated on its own. Expect raw ECE to differ and calibrated ECE to converge.
    option_order  one choice task x model, the same options in several orders (original plus
                  seeded shuffles that each put a different option first). Expect the first
                  option to be over-predicted raw, less so after per-variant calibration.
    transfer      the calibrator selected for task A applied, unchanged, to task B's test split.
                  Expect it to do worse than B's own calibrator.
    cross_model   every model side by side per task, from the main metrics.json files.

Variants run through the same score / calibrate / evaluate code as the main matrix
(cache, budget, mock routing, fingerprints), with one shared budget for the stage.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.core.types import Question, TaskSpec
from recalibench.evaluate import metrics as m
from recalibench.models.budget import Budget
from recalibench.models.context import RunContext
from recalibench.pipeline import scores_io
from recalibench.pipeline.calibrate import calibrate_pair, label_indices, read_selected
from recalibench.pipeline.evaluate import METRICS, calibrated_probs, evaluate_pair, point_metrics
from recalibench.pipeline.score import PairResult, ScorePlan, estimate_and_run, load_task, plan_pair

log = logging.getLogger(__name__)

SUMMARY = "summary.json"


class ExperimentError(Exception):
    """An experiment is misconfigured or its inputs are missing."""


@dataclass
class ExperimentResult:
    name: str
    status: str  # done | skipped | failed | estimated
    reason: str = ""
    summary: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class _Variant:
    experiment: str
    name: str
    spec: TaskSpec
    model_name: str
    out_dir: Path


# ---- variant builders ------------------------------------------------------------


def wording_questions(spec: TaskSpec, n: int) -> list[Question]:
    phrasings = [str(p) for p in cast(list[Any], spec.extra.get("phrasings") or [])]
    texts = [spec.question.text, *phrasings][:n]
    if len(texts) < n:
        log.warning("wording: %s has only %d phrasings, wanted %d", spec.name, len(texts), n)
    return [replace(spec.question, text=t) for t in texts]


def option_orders(question: Question, n: int, seed: int) -> list[Question]:
    """The original order plus seeded shuffles, each putting a not-yet-used option first."""
    if question.kind != "choice":
        raise ExperimentError("option_order needs a choice task")
    rng = np.random.default_rng(seed)
    options = list(question.options)
    out = [question]
    used_first = {options[0].key}
    for _ in range(n - 1):
        candidates = [o for o in options if o.key not in used_first] or options[1:]
        first = candidates[int(rng.integers(len(candidates)))]
        rest = [o for o in options if o.key != first.key]
        perm: list[int] = [int(i) for i in rng.permutation(len(rest)).tolist()]
        order = [first, *[rest[i] for i in perm]]
        used_first.add(first.key)
        out.append(replace(question, options=tuple(order)))
    return out


def _variants(reg: Registry, paths: Paths, cfg: DictConfig) -> list[_Variant]:
    extras = cast(dict[str, Any], OmegaConf.to_container(cfg.experiment.get("extras") or {}))
    out: list[_Variant] = []
    for exp in ("wording", "option_order"):
        settings = extras.get(exp)
        if not settings:
            continue
        spec = reg.task(str(settings["task"]))
        model = reg.model(str(settings["model"]))
        if exp == "wording":
            qs = wording_questions(spec, int(settings.get("phrasings", 3)))
            names = [f"phrasing_{i}" for i in range(len(qs))]
        else:
            qs = option_orders(spec.question, int(settings.get("orders", 3)), int(cfg.seed))
            names = [f"order_{i}" for i in range(len(qs))]
        for name, q in zip(names, qs, strict=True):
            out_dir = paths.experiment_dir(exp, name, model.name, model.version)
            out.append(
                _Variant(exp, name, replace(spec, question=q), str(settings["model"]), out_dir)
            )
    return out


# ---- summaries -------------------------------------------------------------------


def _ci(metrics: dict[str, Any], side: str, name: str) -> dict[str, float]:
    return cast(dict[str, float], metrics[side][name])


def _load_metrics(directory: Path) -> dict[str, Any] | None:
    path = directory / METRICS
    return cast(dict[str, Any], json.loads(path.read_text())) if path.exists() else None


def _wording_summary(variants: list[_Variant]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for v in variants:
        met = _load_metrics(v.out_dir)
        if met is None:
            continue
        rows.append(
            {
                "variant": v.name,
                "question": v.spec.question.text,
                "selected": met["selected"],
                "raw_ece": _ci(met, "raw", "ece"),
                "calibrated_ece": _ci(met, "calibrated", "ece"),
                "raw_log_loss": _ci(met, "raw", "log_loss"),
                "calibrated_log_loss": _ci(met, "calibrated", "log_loss"),
            }
        )

    def spread(key: str) -> float:
        values = [r[key]["value"] for r in rows]
        return float(max(values) - min(values)) if values else float("nan")

    return {
        "variants": rows,
        "raw_ece_spread": spread("raw_ece"),
        "calibrated_ece_spread": spread("calibrated_ece"),
    }


def _option_order_summary(variants: list[_Variant]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mean_probs: dict[str, list[dict[str, float]]] = {"raw": [], "calibrated": []}
    for v in variants:
        met = _load_metrics(v.out_dir)
        sel = read_selected(v.out_dir)
        if met is None or sel is None:
            continue
        scores = scores_io.read_scores(v.out_dir)
        test = (scores.frame["split"] == "test").to_numpy()
        y = label_indices(scores)[test]
        keys = scores.keys
        P = {
            "raw": scores.probs("test"),
            "calibrated": calibrated_probs(v.out_dir, scores)[sel["selected"]][test],
        }
        first = keys[0]  # score columns follow the question's (shuffled) option order
        row: dict[str, Any] = {
            "variant": v.name,
            "order": keys,
            "first_option": first,
            "first_option_true_share": float((y == 0).mean()),
            "selected": met["selected"],
        }
        for side, mat in P.items():
            row[f"{side}_first_option_mean_prob"] = float(mat[:, 0].mean())
            row[f"{side}_first_option_predicted_share"] = float((mat.argmax(axis=1) == 0).mean())
            row[f"{side}_classwise_ece"] = float(m.classwise_ece(mat, y))
            mean_probs[side].append({k: float(mat[:, i].mean()) for i, k in enumerate(keys)})
        rows.append(row)

    def instability(side: str) -> float:
        """Largest move of any option's mean probability across orders."""
        per = mean_probs[side]
        if len(per) < 2:
            return float("nan")
        return float(max(max(d[k] for d in per) - min(d[k] for d in per) for k in per[0]))

    return {
        "variants": rows,
        "raw_mean_prob_instability": instability("raw"),
        "calibrated_mean_prob_instability": instability("calibrated"),
    }


def _transfer(
    reg: Registry, paths: Paths, cfg: DictConfig, settings: dict[str, Any]
) -> ExperimentResult:
    model = reg.model(str(settings["model"]))
    src = paths.artifact_dir(str(settings["from"]), model.name, model.version)
    dst = paths.artifact_dir(str(settings["to"]), model.name, model.version)
    src_sel, dst_sel = read_selected(src), read_selected(dst)
    if src_sel is None or dst_sel is None:
        return ExperimentResult(
            "transfer", "skipped", "both tasks must be calibrated first (recal calibrate)"
        )
    if src_sel["kind"] != dst_sel["kind"] or src_sel["keys"] != dst_sel["keys"]:
        return ExperimentResult("transfer", "skipped", "tasks have different kinds or options")
    with (src / "calibrators" / f"{src_sel['selected']}.pkl").open("rb") as f:
        calibrator = pickle.load(f)  # our own artifact, written by the calibrate stage
    scores = scores_io.read_scores(dst)
    test = (scores.frame["split"] == "test").to_numpy()
    y = label_indices(scores)[test]
    raw = scores.probs("test")
    kind, bins = str(dst_sel["kind"]), int(cfg.ece_bins)
    native = calibrated_probs(dst, scores)[dst_sel["selected"]][test]
    summary = {
        "from": settings["from"],
        "to": settings["to"],
        "model_id": f"{model.name}@{model.version}",
        "transferred_calibrator": src_sel["selected"],
        "native_calibrator": dst_sel["selected"],
        "note": "deliberately applies a calibrator to scores it was not fitted on",
        "raw": point_metrics(raw, y, kind, bins),
        "transferred": point_metrics(
            np.asarray(calibrator.predict(raw), dtype=np.float64), y, kind, bins
        ),
        "native": point_metrics(native, y, kind, bins),
    }
    return ExperimentResult("transfer", "done", summary=summary)


def _cross_model(reg: Registry, paths: Paths, cfg: DictConfig) -> ExperimentResult:
    tasks: dict[str, list[dict[str, Any]]] = {}
    for task in [str(t) for t in cfg.experiment.tasks]:
        rows: list[dict[str, Any]] = []
        for name in [str(x) for x in cfg.experiment.models]:
            model = reg.model(name)
            met = _load_metrics(paths.artifact_dir(task, model.name, model.version))
            if met is None:
                continue
            row: dict[str, Any] = {"model_id": met["model_id"], "selected": met["selected"]}
            for metric in ("ece", "log_loss", "brier", "accuracy", "auroc"):
                if metric in met["raw"]:
                    row[f"raw_{metric}"] = met["raw"][metric]
                    row[f"calibrated_{metric}"] = met["calibrated"][metric]
            row["raw_trust_coverage"] = met["selective"]["raw"]["trust"]["coverage"]
            row["calibrated_trust_coverage"] = met["selective"]["calibrated"]["trust"]["coverage"]
            rows.append(row)
        tasks[task] = rows
    return ExperimentResult("cross_model", "done", summary={"tasks": tasks})


def _write(paths: Paths, result: ExperimentResult) -> None:
    out = paths.experiments_root() / result.name
    out.mkdir(parents=True, exist_ok=True)
    (out / SUMMARY).write_text(json.dumps(result.summary, indent=2, default=float) + "\n")


# ---- stage -----------------------------------------------------------------------


def run_experiments(
    reg: Registry,
    paths: Paths,
    cfg: DictConfig,
    force: bool = False,
    budget: Budget | None = None,
) -> list[ExperimentResult]:
    """`budget`: pass the run's shared Budget; a standalone stage makes its own."""
    if cfg.get("limit") is not None:
        raise ExperimentError("experiments don't support limit=N; smoke-test with `recal score`")
    extras = cast(dict[str, Any], OmegaConf.to_container(cfg.experiment.get("extras") or {}))
    ctx = RunContext(paths=paths, budget=budget or Budget(cap_usd=float(cfg.budget_usd)))
    variants = _variants(reg, paths, cfg)

    plans: list[ScorePlan] = []
    scored: list[PairResult] = []
    loaded: dict[str, tuple[list[Any], list[str], str]] = {}
    for v in variants:
        if v.spec.name not in loaded:
            loaded[v.spec.name] = load_task(paths, v.spec.name, None)
        items, splits, task_fp = loaded[v.spec.name]
        planned = plan_pair(
            reg, ctx, cfg, v.spec, v.model_name, items, splits, task_fp, v.out_dir, force
        )
        if isinstance(planned, PairResult):
            scored.append(planned)
        else:
            plans.append(planned)
    run = estimate_and_run(reg, plans, ctx, cfg)
    if cfg.get("dry_run"):
        total = sum(r.estimate_usd for r in run)
        return [ExperimentResult("estimate", "estimated", f"${total:.4f} for {len(run)} variants")]
    failed = [r for r in scored + run if r.status == "failed"]
    if failed:
        raise ExperimentError("; ".join(f"{r.task}: {r.reason}" for r in failed))

    for v in variants:
        if scores_io.read_meta(v.out_dir) is not None:
            calibrate_pair(reg, v.out_dir, cfg, force)
            evaluate_pair(v.out_dir, cfg, force)

    results: list[ExperimentResult] = []
    by_exp: dict[str, list[_Variant]] = {}
    for v in variants:
        by_exp.setdefault(v.experiment, []).append(v)
    if "wording" in by_exp:
        results.append(
            ExperimentResult("wording", "done", summary=_wording_summary(by_exp["wording"]))
        )
    if "option_order" in by_exp:
        summary = _option_order_summary(by_exp["option_order"])
        results.append(ExperimentResult("option_order", "done", summary=summary))
    if extras.get("transfer"):
        results.append(_transfer(reg, paths, cfg, cast(dict[str, Any], extras["transfer"])))
    results.append(_cross_model(reg, paths, cfg))
    for r in results:
        if r.status == "done":
            _write(paths, r)
    return results
