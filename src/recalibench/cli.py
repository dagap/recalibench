"""`recal` command-line interface.

Commands take Hydra overrides (e.g. `experiment=demo mode=mock`) as positional
arguments. Commands whose phase is not built yet exit 1.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from recalibench.core.config import load_config
from recalibench.core.paths import Paths
from recalibench.core.registry import GROUPS, Group, PluginError, Registry
from recalibench.datasets.manifest import DownloadError
from recalibench.models.base_api import FatalApiError
from recalibench.models.budget import Budget, BudgetExceeded
from recalibench.pipeline import calibrate as calibrate_stage
from recalibench.pipeline import download as download_stage
from recalibench.pipeline import evaluate as evaluate_stage
from recalibench.pipeline import experiments as experiments_stage
from recalibench.pipeline import prepare as prepare_stage
from recalibench.pipeline import report as report_stage
from recalibench.pipeline import score as score_stage
from recalibench.report.collect import ReportData, ReportError
from recalibench.tasks.builder import BuildError

app = typer.Typer(
    name="recal",
    help="Recalibrate probabilistic decision models on labelled datasets.",
    no_args_is_help=True,
)


@dataclass
class State:
    config_dirs: list[Path]


@app.callback()
def main(
    ctx: typer.Context,
    config_dir: Annotated[
        list[Path] | None,
        typer.Option(
            "--config-dir",
            help="Extra Hydra config dir (repeatable); its plug-ins merge with the built-in ones.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    ctx.obj = State(config_dirs=config_dir or [])


Overrides = Annotated[
    list[str] | None,
    typer.Argument(help="Hydra overrides, e.g. experiment=demo mode=mock."),
]


def _describe(reg: Registry, group: Group, name: str) -> str:
    if group == "task":
        t = reg.task(name)
        return f"dataset={t.dataset} kind={t.question.kind}"
    target = reg.target(group, name)
    if group == "dataset":
        reg.dataset(name)
        return f"{target}"
    if group == "model":
        m = reg.model(name)
        return (
            f"{m.name}@{m.version} supports={sorted(m.supports)} trainable={m.trainable} ({target})"
        )
    c = reg.calibrator(name)
    return f"supports={sorted(c.supports)} ({target})"


@app.command("list")
def list_plugins(ctx: typer.Context, overrides: Overrides = None) -> None:
    """List registered datasets, tasks, models and calibrators."""
    state: State = ctx.obj
    reg = Registry(extra_conf_dirs=state.config_dirs, overrides=overrides or [])
    failed = False
    for group in GROUPS:
        typer.echo(f"{group}s:")
        names = reg.names(group)
        if not names:
            typer.echo("  (none)")
        for name in names:
            try:
                typer.echo(f"  {name:<20} {_describe(reg, group, name)}")
            except PluginError as e:
                failed = True
                typer.echo(f"  {name:<20} ERROR: {e}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def download(
    ctx: typer.Context,
    overrides: Overrides = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", help="Dataset to fetch (repeatable). Default: the experiment's."),
    ] = None,
) -> None:
    """Download raw data for the experiment's datasets (or --dataset ones)."""
    state: State = ctx.obj
    cfg = load_config(overrides or [], extra_conf_dirs=state.config_dirs)
    reg = Registry(extra_conf_dirs=state.config_dirs, overrides=overrides or [])
    try:
        names = download_stage.datasets_for(reg, cfg, dataset or [])
    except PluginError as e:
        typer.echo(
            f"{e}\nFix the experiment's tasks, or pick datasets with --dataset NAME.", err=True
        )
        raise typer.Exit(code=1) from e
    try:
        download_stage.download(reg, Paths.from_config(cfg), names)
    except DownloadError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e
    except (PluginError, download_stage.SchemaError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e


@app.command()
def prepare(
    ctx: typer.Context,
    overrides: Overrides = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if up to date.")] = False,
) -> None:
    """Build items and fit/val/test splits for each of the experiment's tasks."""
    state: State = ctx.obj
    cfg = load_config(overrides or [], extra_conf_dirs=state.config_dirs)
    reg = Registry(extra_conf_dirs=state.config_dirs, overrides=overrides or [])
    try:
        prepare_stage.prepare(reg, Paths.from_config(cfg), cfg, force=force)
    except (PluginError, prepare_stage.PrepareError, BuildError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e


STAGE_ERRORS = (
    PluginError,
    DownloadError,
    prepare_stage.PrepareError,
    BuildError,
    score_stage.ScoreError,
    BudgetExceeded,
    FatalApiError,
    evaluate_stage.StaleCalibration,
    experiments_stage.ExperimentError,
    ReportError,
)

Force = Annotated[bool, typer.Option("--force", help="Redo work even if up to date.")]


def _setup(ctx: typer.Context, overrides: list[str] | None) -> tuple[Registry, Paths, DictConfig]:
    state: State = ctx.obj
    cfg = load_config(overrides or [], extra_conf_dirs=state.config_dirs)
    reg = Registry(extra_conf_dirs=state.config_dirs, overrides=overrides or [])
    return reg, Paths.from_config(cfg), cfg


def _report_scores(results: list[score_stage.PairResult]) -> None:
    for r in results:
        if r.status == "estimated":
            detail = f"estimate ${r.estimate_usd:.4f}"
        else:
            detail = r.reason or f"{r.rows} rows, {r.errors} errors, ${r.cost_usd:.4f}"
        typer.echo(f"  {r.status:<9} {r.task} x {r.model}: {detail}")
    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=1)


@app.command()
def score(ctx: typer.Context, overrides: Overrides = None, force: Force = False) -> None:
    """Score every task x model (cached, budgeted). dry_run=true estimates; limit=N smoke-tests."""
    reg, paths, cfg = _setup(ctx, overrides)
    try:
        results = score_stage.score(reg, paths, cfg, force=force)
    except STAGE_ERRORS as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    _report_scores(results)


def _report_calibration(results: list[calibrate_stage.CalibrationResult]) -> None:
    for r in results:
        if r.status == "calibrated":
            detail = (
                f"selected {r.selected} (val log loss "
                f"{r.raw_val_log_loss:.4f} -> {r.selected_val_log_loss:.4f})"
            )
        else:
            detail = r.reason or f"selected {r.selected}"
        typer.echo(f"  {r.status:<10} {r.task} x {r.model}: {detail}")
    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=1)


@app.command()
def calibrate(ctx: typer.Context, overrides: Overrides = None, force: Force = False) -> None:
    """Fit every compatible calibrator on the fit split and select one on val."""
    reg, paths, cfg = _setup(ctx, overrides)
    try:
        results = calibrate_stage.calibrate(reg, paths, cfg, force=force)
    except STAGE_ERRORS as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    _report_calibration(results)


def _report_evaluation(results: list[evaluate_stage.EvalResult]) -> None:
    for r in results:
        if r.status == "evaluated":
            detail = f"test ECE {r.raw_ece:.4f} -> {r.selected_ece:.4f} ({r.selected})"
        else:
            detail = r.reason or f"selected {r.selected}"
        typer.echo(f"  {r.status:<10} {r.task} x {r.model}: {detail}")
    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=1)


@app.command()
def evaluate(ctx: typer.Context, overrides: Overrides = None, force: Force = False) -> None:
    """Test-split metrics with paired-bootstrap CIs, selective prediction -> metrics.json."""
    reg, paths, cfg = _setup(ctx, overrides)
    try:
        results = evaluate_stage.evaluate(reg, paths, cfg, force=force)
    except STAGE_ERRORS as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    _report_evaluation(results)


def _headline(r: experiments_stage.ExperimentResult) -> str:
    s = r.summary
    if r.name == "wording":
        raw, cal = s["raw_ece_spread"], s["calibrated_ece_spread"]
        return f"raw ECE spread across phrasings {raw:.4f} -> calibrated {cal:.4f}"
    if r.name == "option_order":
        return (
            f"mean-probability instability {s['raw_mean_prob_instability']:.4f} -> "
            f"{s['calibrated_mean_prob_instability']:.4f}"
        )
    if r.name == "transfer":
        return (
            f"{s['from']} -> {s['to']}: ECE raw {s['raw']['ece']:.4f}, transferred "
            f"{s['transferred']['ece']:.4f}, native {s['native']['ece']:.4f}"
        )
    if r.name == "cross_model":
        return f"{sum(len(v) for v in s['tasks'].values())} task x model rows"
    return r.reason


@app.command()
def experiments(ctx: typer.Context, overrides: Overrides = None, force: Force = False) -> None:
    """Run the wording, option-order, transfer and cross-model experiments (experiment.extras)."""
    reg, paths, cfg = _setup(ctx, overrides)
    try:
        results = experiments_stage.run_experiments(reg, paths, cfg, force=force)
    except STAGE_ERRORS as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    _report_experiments(results)


def _report_experiments(results: list[experiments_stage.ExperimentResult]) -> None:
    for r in results:
        typer.echo(f"  {r.status:<9} {r.name}: {_headline(r) if r.status == 'done' else r.reason}")


def _report_built(path: Path, data: ReportData) -> None:
    typer.echo(
        f"  built     {path} ({len(data.pairs)} task x model pairs, "
        f"{len(data.experiments)} experiments)"
    )


@app.command()
def report(ctx: typer.Context, overrides: Overrides = None) -> None:
    """Build reports/report.html (live) or reports/report_mock.html from artifacts."""
    _, paths, cfg = _setup(ctx, overrides)
    try:
        path, data = report_stage.build_report(paths, cfg)
    except STAGE_ERRORS as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    _report_built(path, data)


@app.command()
def run(ctx: typer.Context, overrides: Overrides = None, force: Force = False) -> None:
    """Run the pipeline end to end: download, prepare, score, calibrate, evaluate,
    experiments, report."""
    reg, paths, cfg = _setup(ctx, overrides)
    budget = Budget(cap_usd=float(cfg.budget_usd))  # one per run, shared by every stage
    try:  # each stage reports as it finishes, so a later failure keeps earlier summaries
        download_stage.download(reg, paths, download_stage.datasets_for(reg, cfg))
        prepare_stage.prepare(reg, paths, cfg, force=force)
        _report_scores(score_stage.score(reg, paths, cfg, force=force, budget=budget))
        _report_calibration(calibrate_stage.calibrate(reg, paths, cfg, force=force))
        _report_evaluation(evaluate_stage.evaluate(reg, paths, cfg, force=force))
        if cfg.get("limit") is not None:
            typer.echo("  skipped   experiments and report: limit=N is a smoke run")
            return
        _report_experiments(
            experiments_stage.run_experiments(reg, paths, cfg, force=force, budget=budget)
        )
        _report_built(*report_stage.build_report(paths, cfg))
    except STAGE_ERRORS as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2 if isinstance(e, DownloadError) else 1) from e
