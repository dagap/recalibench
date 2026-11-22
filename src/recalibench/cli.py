"""`recal` command-line interface.

Commands take Hydra overrides (e.g. `experiment=demo mode=mock`) as positional
arguments. Commands whose phase is not built yet exit 1.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from recalibench.core.config import load_config
from recalibench.core.paths import Paths
from recalibench.core.registry import GROUPS, Group, PluginError, Registry
from recalibench.datasets.manifest import DownloadError
from recalibench.pipeline import download as download_stage
from recalibench.pipeline import prepare as prepare_stage
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


def _not_implemented(command: str, phase: int) -> None:
    typer.echo(f"recal {command}: not implemented yet (Phase {phase}).", err=True)
    raise typer.Exit(code=1)


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


@app.command()
def score(overrides: Overrides = None) -> None:
    """Score every task x model (cached, budgeted)."""
    _not_implemented("score", 5)


@app.command()
def calibrate(overrides: Overrides = None) -> None:
    """Fit calibrators on the fit split and select on val."""
    _not_implemented("calibrate", 6)


@app.command()
def evaluate(overrides: Overrides = None) -> None:
    """Compute test metrics with bootstrap CIs."""
    _not_implemented("evaluate", 7)


@app.command()
def experiments(overrides: Overrides = None) -> None:
    """Run the wording, transfer and cross-model experiments."""
    _not_implemented("experiments", 7)


@app.command()
def report(overrides: Overrides = None) -> None:
    """Build reports/report.html from artifacts."""
    _not_implemented("report", 8)


@app.command()
def run(overrides: Overrides = None) -> None:
    """Run the full pipeline end to end."""
    _not_implemented("run", 5)
