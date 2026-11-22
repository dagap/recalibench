"""`recal` command-line interface.

Phase 0 stub: every command exists so the Makefile and `--help` show the full
surface, but each exits 1 until its phase implements it. Commands take Hydra
overrides (e.g. `experiment=demo mode=mock`) as positional arguments.
"""

from typing import Annotated

import typer

app = typer.Typer(
    name="recal",
    help="Recalibrate probabilistic decision models on labelled datasets.",
    no_args_is_help=True,
)

Overrides = Annotated[
    list[str] | None,
    typer.Argument(help="Hydra overrides, e.g. experiment=demo mode=mock."),
]


def _not_implemented(command: str, phase: int) -> None:
    typer.echo(f"recal {command}: not implemented yet (Phase {phase}).", err=True)
    raise typer.Exit(code=1)


@app.command("list")
def list_plugins(overrides: Overrides = None) -> None:
    """List registered datasets, tasks, models and calibrators."""
    _not_implemented("list", 1)


@app.command()
def download(overrides: Overrides = None) -> None:
    """Download raw data for the experiment's datasets."""
    _not_implemented("download", 2)


@app.command()
def prepare(overrides: Overrides = None) -> None:
    """Build items and fit/val/test splits for each task."""
    _not_implemented("prepare", 3)


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
