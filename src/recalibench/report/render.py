"""Render ReportData as one self-contained HTML document (inline CSS and SVG only).

Sections come from loops over what collect() found: no task, model or calibrator is
named here. Experiments have a renderer per experiment kind, and anything unknown or
partial falls back to a generic table. Every string from artifacts is escaped.
"""

import html
import json
from collections.abc import Callable
from typing import Any

from recalibench.report import figures
from recalibench.report.collect import Pair, ReportData, ci

METRICS = [
    "ece",
    "ece_equal_mass",
    "classwise_ece",
    "smooth_ece",
    "brier",
    "log_loss",
    "accuracy",
    "auroc",
]
DASH = "—"


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _num(x: float | None, digits: int = 4) -> str:
    return DASH if x is None else f"{x:.{digits}f}"


def _ci_cell(metric: dict[str, Any] | None) -> str:
    c = ci(metric)
    if c is None:
        return DASH
    value, low, high = c
    bounds = (
        ""
        if low is None or high is None
        else f' <span class="ci">[{_num(low)}, {_num(high)}]</span>'
    )
    return f"{_num(value)}{bounds}"


def _ece(pair: Pair, side: str) -> float | None:
    c = ci(pair.metrics[side].get("ece"))
    return None if c is None else c[0]


def raw_already_better(pair: Pair) -> bool:
    raw, cal = _ece(pair, "raw"), _ece(pair, "calibrated")
    return raw is not None and cal is not None and cal >= raw


def _table(head: list[str], rows: list[list[str]]) -> str:
    """Cells are already-escaped HTML; the first cell of each row is a row header."""
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(
        f'<tr><th scope="row">{r[0]}</th>' + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>"
        for r in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _figs(*svgs: str) -> str:
    return '<div class="figs">' + "".join(f"<figure>{s}</figure>" for s in svgs) + "</div>"


# ---- summary ---------------------------------------------------------------------------


def _summary(data: ReportData) -> str:
    by_key = {(p.task, p.model_id): p for p in data.pairs}
    rows: list[list[str]] = []
    for task in data.tasks:
        cells = [f'<a href="#task-{e(task)}">{e(task)}</a>']
        for model in data.models:
            p = by_key.get((task, model))
            if p is None:
                cells.append(DASH)
                continue
            cell = (
                f'<a href="#pair-{e(task)}-{e(model)}">{_num(_ece(p, "raw"), 3)} → '
                f'{_num(_ece(p, "calibrated"), 3)}</a> <span class="sel">({e(p.selected)})</span>'
            )
            if raw_already_better(p):
                cell += ' <span class="flag">raw already better</span>'
            cells.append(cell)
        rows.append(cells)
    out = (
        '<section id="summary"><h2>Summary: test ECE, raw → calibrated</h2>'
        "<p>Top-label expected calibration error on the test split (lower is better), "
        "before and after the calibrator selected on the validation split.</p>"
        + _table(["task"] + [e(m) for m in data.models], rows)
    )
    if data.not_evaluated:
        items = "".join(f"<li>{e(t)} × {e(m)}</li>" for t, m in data.not_evaluated)
        out += (
            "<p>Not evaluated yet (no metrics.json; run <code>recal evaluate</code>):</p>"
            f"<ul>{items}</ul>"
        )
    return out + "</section>"


# ---- per task and pair -------------------------------------------------------------------


def _metrics_table(pair: Pair) -> str:
    m = pair.metrics
    rows = [
        [
            e(name),
            _ci_cell(m["raw"].get(name)),
            _ci_cell(m["calibrated"].get(name)),
            _ci_cell(m.get("difference", {}).get(name)),
        ]
        for name in METRICS
    ]
    return _table(["metric", "raw", e(f"calibrated ({pair.selected})"), "difference"], rows)


def _selective_table(pair: Pair) -> str:
    sel = pair.metrics["selective"]
    raw, cal = sel["raw"], sel["calibrated"]
    target = float(raw.get("target_accuracy", 0.95))

    def pct(x: object) -> str:
        return f"{float(str(x)):.0%}"

    rows = [
        [
            f"tuned (threshold set on val for {target:.0%} accuracy)",
            f"{pct(raw['tuned']['coverage'])} at {pct(raw['tuned']['accuracy'])}",
            f"{pct(cal['tuned']['coverage'])} at {pct(cal['tuned']['accuracy'])}",
        ],
        [
            f"trust the number (confidence ≥ {float(raw['trust']['threshold']):.2f})",
            f"{pct(raw['trust']['coverage'])} at {pct(raw['trust']['accuracy'])}",
            f"{pct(cal['trust']['coverage'])} at {pct(cal['trust']['accuracy'])}",
        ],
        ["AURC (lower is better)", _num(float(raw["aurc"])), _num(float(cal["aurc"]))],
    ]
    return _table(["auto-handled share, at accuracy", "raw", "calibrated"], rows)


def _pair_section(pair: Pair) -> str:
    raw, cal = _ece(pair, "raw"), _ece(pair, "calibrated")
    verdict = (
        '<p class="flag-line">Raw already better: the selected calibrator did not lower '
        "test ECE here.</p>"
        if raw_already_better(pair)
        else ""
    )
    return (
        f'<section id="pair-{e(pair.task)}-{e(pair.model_id)}" class="pair">'
        f"<h3>{e(pair.model_id)}</h3>"
        f"<p>Selected calibrator <b>{e(pair.selected)}</b>; test ECE {_num(raw)} → "
        f"{_num(cal)} on {pair.n_test} test rows.</p>{verdict}"
        + _metrics_table(pair)
        + _selective_table(pair)
        + _figs(
            figures.reliability_svg(pair),
            figures.confidence_hist_svg(pair),
            figures.risk_coverage_svg(pair),
        )
        + "</section>"
    )


def _task_section(task: str, pairs: list[Pair]) -> str:
    first = pairs[0]
    return (
        f'<section id="task-{e(task)}" class="task"><h2>{e(task)}</h2>'
        f"<p>{e(first.kind)} task on dataset <b>{e(first.dataset)}</b>, "
        f"{len(pairs)} model(s).</p>"
        + _figs(figures.cross_model_ece_svg(task, pairs))
        + "".join(_pair_section(p) for p in pairs)
        + "</section>"
    )


# ---- experiments -------------------------------------------------------------------------


def _wording(s: dict[str, Any]) -> str:
    rows = [
        [
            e(v["variant"]),
            e(v["question"]),
            e(v["selected"]),
            _ci_cell(v["raw_ece"]),
            _ci_cell(v["calibrated_ece"]),
        ]
        for v in s["variants"]
    ]
    return (
        f"<p>Spread of test ECE across phrasings: raw {_num(s['raw_ece_spread'])}, "
        f"calibrated {_num(s['calibrated_ece_spread'])}.</p>"
        + _figs(figures.wording_svg(s))
        + _table(["phrasing", "question", "selected", "raw ECE", "calibrated ECE"], rows)
    )


def _option_order(s: dict[str, Any]) -> str:
    rows = [
        [
            e(v["variant"]),
            e(", ".join(str(o) for o in v["order"])),
            e(v["first_option"]),
            _num(v["first_option_true_share"], 3),
            _num(v["raw_first_option_mean_prob"], 3),
            _num(v["calibrated_first_option_mean_prob"], 3),
        ]
        for v in s["variants"]
    ]
    return (
        f"<p>Instability of mean class probabilities across orders: raw "
        f"{_num(s['raw_mean_prob_instability'])}, calibrated "
        f"{_num(s['calibrated_mean_prob_instability'])}.</p>"
        + _figs(figures.option_order_svg(s))
        + _table(
            ["order", "options", "first", "true share", "raw mean prob", "calibrated mean prob"],
            rows,
        )
    )


def _transfer(s: dict[str, Any]) -> str:
    rows = [
        [e(f"{label} ({s[cal_key]})" if cal_key else label)]
        + [_num(s[side].get(name)) for name in METRICS]
        for side, label, cal_key in (
            ("raw", "raw", None),
            ("transferred", "transferred", "transferred_calibrator"),
            ("native", "native", "native_calibrator"),
        )
    ]
    return (
        f"<p>Calibrator fitted on <b>{e(s['from'])}</b> applied to <b>{e(s['to'])}</b> "
        f"({e(s['model_id'])}). {e(s.get('note', ''))}</p>"
        + _table(["test metrics on " + e(s["to"])] + [e(n) for n in METRICS], rows)
    )


def _generic(s: dict[str, Any]) -> str:
    rows = [
        [e(k), f"<code>{e(v if isinstance(v, (str, int, float, bool)) else json.dumps(v))}</code>"]
        for k, v in s.items()
    ]
    return _table(["field", "value"], rows)


RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "wording": _wording,
    "option_order": _option_order,
    "transfer": _transfer,
}
SKIP = {"cross_model"}  # the per-task ECE charts already show it
TITLES = {
    "wording": "Wording: the same question, phrased differently",
    "option_order": "Option order: the same options, listed in different orders",
    "transfer": "Transfer: a calibrator fitted on one task, used on another",
}


def _experiment(name: str, s: dict[str, Any]) -> str:
    renderer = RENDERERS.get(name)
    body: str
    try:
        body = renderer(s) if renderer else _generic(s)
    except (KeyError, TypeError, ValueError, IndexError):
        body = '<p class="note">Partial summary; raw fields:</p>' + _generic(s)
    title = e(TITLES.get(name, name))
    return f'<div class="experiment" id="exp-{e(name)}"><h3>{title}</h3>{body}</div>'


def _experiments(data: ReportData) -> str:
    shown = [(n, s) for n, s in data.experiments.items() if n not in SKIP]
    body = "".join(_experiment(n, s) for n, s in shown) or "<p>No experiments in artifacts.</p>"
    return f'<section id="experiments"><h2>Experiments</h2>{body}</section>'


# ---- cost and config ---------------------------------------------------------------------


def _cost(data: ReportData) -> str:
    rows: list[list[str]] = []
    for model in data.models:
        ps = [p for p in data.pairs if p.model_id == model]
        cost = sum(float(p.metrics.get("cost_usd", 0.0)) for p in ps)
        lat = [float(p.metrics.get("mean_latency_ms", 0.0)) for p in ps]
        lat = [x for x in lat if x > 0]
        rows.append(
            [
                e(model),
                f"${cost:.4f}",
                f"{sum(lat) / len(lat):.0f} ms" if lat else DASH,
                str(len(ps)),
            ]
        )
    note = " Mock mode: answers are simulated and cost nothing." if data.mode == "mock" else ""
    return (
        '<section id="cost"><h2>Cost and latency per model</h2>'
        f"<p>Spend recorded with the scores (cached answers count once).{note}</p>"
        + _table(["model", "cost", "mean latency per item", "tasks"], rows)
        + "</section>"
    )


def _config(run_config: str) -> str:
    return f'<section id="config"><h2>Run config</h2><pre>{e(run_config)}</pre></section>'


CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1f; --muted:#6b6b70; --card:#ffffff; --line:#e3e3e0;
  --accent:#1f5fbf; --flag-bg:#fdf0d5; --flag-fg:#7a4b00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#151517; --fg:#ececee; --muted:#a0a0a8; --card:#1f1f23; --line:#33333a;
  --accent:#7aa7ff; --flag-bg:#4a3a12; --flag-fg:#ffd98a; } }
:root[data-theme="dark"] { --bg:#151517; --fg:#ececee; --muted:#a0a0a8; --card:#1f1f23;
  --line:#33333a; --accent:#7aa7ff; --flag-bg:#4a3a12; --flag-fg:#ffd98a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 64px; }
header h1 { margin:0 0 4px; font-size:28px; letter-spacing:-0.01em; }
.meta { color:var(--muted); margin:0 0 8px; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
  font-weight:600; background:var(--flag-bg); color:var(--flag-fg); }
.badge.live { background:var(--accent); color:#fff; }
nav { margin:16px 0 8px; display:flex; flex-wrap:wrap; gap:6px 14px; font-size:14px; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
h2 { margin:40px 0 8px; font-size:21px; border-top:1px solid var(--line); padding-top:24px; }
h3 { margin:28px 0 6px; font-size:17px; }
p { margin:6px 0 12px; max-width:80ch; }
.scroll { overflow-x:auto; margin:8px 0 16px; }
table { border-collapse:collapse; font-size:13.5px; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:5px 12px 5px 0; border-bottom:1px solid var(--line);
  vertical-align:top; }
thead th { color:var(--muted); font-weight:600; }
tbody th { font-weight:500; }
.ci, .sel, .note { color:var(--muted); font-size:12px; }
.flag { display:inline-block; padding:0 6px; border-radius:4px; font-size:12px;
  background:var(--flag-bg); color:var(--flag-fg); }
.flag-line { background:var(--flag-bg); color:var(--flag-fg); padding:6px 10px;
  border-radius:6px; display:inline-block; }
.figs { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));
  gap:12px; margin:8px 0 16px; }
figure { margin:0; background:#fff; border:1px solid var(--line); border-radius:8px;
  padding:8px; }
figure svg { display:block; width:100%; height:auto; }
pre { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:12px; overflow-x:auto; font-size:12.5px; }
code { font-size:12.5px; word-break:break-word; }
.pair { border-left:3px solid var(--line); padding-left:14px; }
"""


def render(data: ReportData, run_config: str, generated_at: str) -> str:
    by_task: dict[str, list[Pair]] = {}
    for p in data.pairs:
        by_task.setdefault(p.task, []).append(p)
    badge = (
        '<span class="badge">MOCK — simulated model answers, $0</span>'
        if data.mode == "mock"
        else '<span class="badge live">LIVE</span>'
    )
    nav = "".join(f'<a href="#task-{e(t)}">{e(t)}</a>' for t in data.tasks)
    nav += (
        '<a href="#experiments">experiments</a><a href="#cost">cost</a><a href="#config">config</a>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>recalibench report</title><style>{CSS}</style></head><body><main>"
        f"<header><h1>recalibench report</h1>"
        f'<p class="meta">{badge} Generated {e(generated_at)} · {len(data.pairs)} task × model '
        f"pairs · {len(data.tasks)} tasks · {len(data.models)} models</p><nav>{nav}</nav></header>"
        + _summary(data)
        + "".join(_task_section(t, by_task[t]) for t in data.tasks)
        + _experiments(data)
        + _cost(data)
        + _config(run_config)
        + "</main></body></html>"
    )
