"""Report figures as inline SVG strings (matplotlib, Agg). All matplotlib calls live here.

Every plotted series carries a gid ("raw", "calibrated", ...), which matplotlib writes
as an SVG id, so tests can check that a figure really contains data. Figures read only
the parsed metrics.json / summary.json; none knows a task or model by name.

matplotlib types every Axes method's **kwargs as Unknown, so that one pyright rule is
off for this file (the adapter file, as with sklearn), not line by line.
"""
# pyright: reportUnknownMemberType=false

import io
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from recalibench.report.collect import Pair, ci  # noqa: E402

plt.rcParams.update(
    {
        "svg.fonttype": "none",  # text stays text: smaller, searchable
        "svg.hashsalt": "recalibench",  # deterministic internal ids
        "font.size": 8,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
    }
)

RAW = "#b5655f"  # muted red: before calibration
CAL = "#1f5fbf"  # strong blue: the selected calibrator
NEUTRAL = "#9a9a9a"
DEFAULT_BINS = 15  # metrics.json from EVALUATE_VERSION 1 has no reliability.bins
SIZE = (4.2, 3.2)


def _to_svg(fig: Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg") :]


def error_lengths(value: float, low: float | None, high: float | None) -> tuple[float, float]:
    """Error-bar lengths below / above `value`, clipped at 0.

    Bootstrap ECE can sit wholly above its point estimate (resampling inflates ECE),
    and matplotlib rejects negative lengths.
    """
    if low is None or high is None:
        return 0.0, 0.0
    return max(0.0, value - low), max(0.0, high - value)


def _bins(pair: Pair) -> int:
    return int(pair.metrics["reliability"].get("bins", DEFAULT_BINS))


def _top_label(pair: Pair, side: str) -> list[dict[str, Any]]:
    return list(pair.metrics["reliability"][side]["top_label"])


def _counts(pair: Pair, side: str) -> list[float]:
    counts = [0.0] * _bins(pair)
    for b in _top_label(pair, side):
        counts[int(b["bin"])] = float(b["count"])
    return counts


def _edges(n: int) -> list[float]:
    return [i / n for i in range(n + 1)]


def _legend(ax: Axes) -> None:
    ax.legend(frameon=False, loc="best")


def reliability_svg(pair: Pair) -> str:
    """Top-label reliability, raw vs the selected calibrator, with per-bin test counts."""
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=SIZE, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=NEUTRAL, gid="diagonal")
    for side, color, label in (("raw", RAW, "raw"), ("calibrated", CAL, pair.selected)):
        bins = _top_label(pair, side)
        ax.plot(
            [b["confidence"] for b in bins],
            [b["observed"] for b in bins],
            marker="o",
            ms=3,
            lw=1.2,
            color=color,
            label=label,
            gid=side,
        )
        axc.stairs(_counts(pair, side), _edges(_bins(pair)), color=color, lw=1)
    axc.set_gid("counts")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability (top-label)", loc="left")
    _legend(ax)
    axc.set_ylabel("test rows")
    axc.set_xlabel("confidence")
    return _to_svg(fig)


def confidence_hist_svg(pair: Pair) -> str:
    """How confident the model is on test rows, raw vs calibrated (top-label bins)."""
    fig, ax = plt.subplots(figsize=SIZE)
    edges = _edges(_bins(pair))
    ax.stairs(_counts(pair, "raw"), edges, fill=True, alpha=0.35, color=RAW, label="raw", gid="raw")
    ax.stairs(
        _counts(pair, "calibrated"), edges, lw=1.4, color=CAL, label=pair.selected, gid="calibrated"
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("top-label confidence")
    ax.set_ylabel("test rows")
    ax.set_title("Confidence histogram", loc="left")
    _legend(ax)
    return _to_svg(fig)


def _aurc(x: object) -> str:
    return "—" if x is None else f"{float(str(x)):.3f}"


def risk_coverage_svg(pair: Pair) -> str:
    """Error rate among auto-handled cases as coverage grows; lower is better."""
    fig, ax = plt.subplots(figsize=SIZE)
    sel = pair.metrics["selective"]
    for side, color, label in (("raw", RAW, "raw"), ("calibrated", CAL, pair.selected)):
        s = sel[side]
        ax.plot(
            s["curve"]["coverage"],
            s["curve"]["risk"],
            lw=1.4,
            color=color,
            label=f"{label} (AURC {_aurc(s.get('aurc'))})",
            gid=side,
        )
    target = 1 - float(sel["raw"].get("target_accuracy", 0.95))
    ax.axhline(target, ls="--", lw=0.8, color=NEUTRAL)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("coverage (share auto-handled)")
    ax.set_ylabel("error rate")
    ax.set_title("Risk vs coverage", loc="left")
    _legend(ax)
    return _to_svg(fig)


CI = tuple[float, float | None, float | None]


def _ci_rows(labels: Sequence[str], raw: Sequence[CI], cal: Sequence[CI], title: str) -> str:
    """One row per label: raw and calibrated point estimates with clipped CI bars."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(SIZE[0] + 1, 0.9 + 0.45 * n))
    ys = list(range(n))
    for side, rows, color, dy in (("raw", raw, RAW, 0.12), ("calibrated", cal, CAL, -0.12)):
        lengths = [error_lengths(*r) for r in rows]
        ax.errorbar(
            [r[0] for r in rows],
            [y + dy for y in ys],
            xerr=[[lo for lo, _ in lengths], [hi for _, hi in lengths]],
            fmt="o",
            ms=4,
            capsize=2,
            lw=1,
            color=color,
            label=side,
            gid=side,
        )
    ax.set_yticks(ys, list(labels))
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("ECE (95% CI)")
    ax.set_title(title, loc="left", pad=18)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)  # above rows
    return _to_svg(fig)


def _point(metric: dict[str, Any] | None) -> CI:
    return ci(metric) or (float("nan"), None, None)


def cross_model_ece_svg(task: str, pairs: list[Pair]) -> str:
    """Every model on one task: raw vs calibrated test ECE with CIs."""
    return _ci_rows(
        [p.model_id for p in pairs],
        [_point(p.metrics["raw"].get("ece")) for p in pairs],
        [_point(p.metrics["calibrated"].get("ece")) for p in pairs],
        f"ECE by model: {task}",
    )


def wording_svg(summary: dict[str, Any]) -> str:
    variants = list(summary["variants"])
    return _ci_rows(
        [str(v["variant"]) for v in variants],
        [_point(v["raw_ece"]) for v in variants],
        [_point(v["calibrated_ece"]) for v in variants],
        "ECE by question wording",
    )


def option_order_svg(summary: dict[str, Any]) -> str:
    """Mean probability of the first-listed option, per order, against its true share."""
    variants = list(summary["variants"])
    xs = list(range(len(variants)))
    fig, ax = plt.subplots(figsize=SIZE)
    ax.plot(
        xs,
        [float(v["first_option_true_share"]) for v in variants],
        ls="none",
        marker="_",
        ms=18,
        mew=2,
        color=NEUTRAL,
        label="true share",
        gid="true_share",
    )
    for side, color, marker in (("raw", RAW, "o"), ("calibrated", CAL, "s")):
        ax.plot(
            xs,
            [float(v[f"{side}_first_option_mean_prob"]) for v in variants],
            ls="none",
            marker=marker,
            ms=5,
            color=color,
            label=side,
            gid=side,
        )
    ax.set_xticks(xs, [f"{v['variant']}\n({v['first_option']} first)" for v in variants])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("mean probability of the first option")
    ax.set_title("First-listed option, by option order", loc="left")
    _legend(ax)
    return _to_svg(fig)
