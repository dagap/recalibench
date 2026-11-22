"""recalibench demo: is the model's confidence honest, and what does fixing it buy?

Run with `make demo`. Reads only what the pipeline wrote (artifacts*/, data/processed/);
the artifacts root comes from $RECAL_ROOT (default: the current directory).
"""

import os
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from recalibench.app import data, live
from recalibench.core.paths import Mode, Paths
from recalibench.models.budget import Budget, BudgetExceeded
from recalibench.report.collect import Pair, ReportData, ci

RAW, CAL = "#b5655f", "#1f5fbf"
COLORS = alt.Scale(domain=["raw", "calibrated"], range=[RAW, CAL])

st.set_page_config(page_title="recalibench", layout="wide")
ROOT = Path(os.environ.get("RECAL_ROOT", ".")).resolve()


# ---- loading (cached by file modification time, so a re-run of the pipeline shows) ----


def _stamp(paths: Paths) -> float:
    files = list(paths.artifacts_root().glob("*/*/metrics.json"))
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner=False)
def _load(root: str, mode: Mode, stamp: float) -> ReportData:
    return data.load(Paths(Path(root), mode))


@st.cache_data(show_spinner=False)
def _rows(root: str, mode: Mode, task: str, model_id: str, stamp: float) -> pd.DataFrame:
    paths = Paths(Path(root), mode)
    pair = next(p for p in data.load(paths).pairs if (p.task, p.model_id) == (task, model_id))
    return data.test_rows(paths, pair)


def _fmt_ci(metric: dict[str, Any] | None) -> str:
    c = ci(metric)
    if c is None:
        return "—"
    value, low, high = c
    return f"{value:.3f}" if low is None or high is None else f"{value:.3f} [{low:.3f}, {high:.3f}]"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


# ---- sidebar -----------------------------------------------------------------------

modes = data.available_modes(ROOT)
if not modes:
    st.title("recalibench")
    st.error(f"No evaluated results under {ROOT}. Build them with `make all` (mock, free) first.")
    st.stop()

with st.sidebar:
    st.title("recalibench")
    mode: Mode = st.radio(
        "Results",
        modes,
        format_func=lambda m: (
            "Live (real model answers)" if m == "live" else "Mock (simulated, $0)"
        ),
        key="mode",
    )
    paths = Paths(ROOT, mode)
    stamp = _stamp(paths)
    report = _load(str(ROOT), mode, stamp)
    task: str = st.selectbox("Task", report.tasks, key="task")
    task_models = [p.model_id for p in report.pairs if p.task == task]
    model_id: str = st.selectbox("Model", task_models, key="model")
    if mode == "mock":
        st.warning("Mock mode: hosted-model answers are simulated, not real.")

pair: Pair = next(p for p in report.pairs if (p.task, p.model_id) == (task, model_id))
m = pair.metrics

overview, reliability, decision, examples = st.tabs(
    ["Overview", "Reliability", "Decision policy", "Examples"]
)

# ---- Overview ------------------------------------------------------------------------

with overview:
    st.header("Is the confidence honest, before and after calibration?")
    st.caption(
        "ECE (expected calibration error): how far stated confidence is from actual accuracy, "
        "on the held-out test split. Lower is better; 95% bootstrap intervals in brackets."
    )
    rows: list[dict[str, Any]] = []
    for p in report.pairs:
        raw_e, cal_e = ci(p.metrics["raw"].get("ece")), ci(p.metrics["calibrated"].get("ece"))
        trust = p.metrics.get("selective", {})
        rows.append(
            {
                "task": p.task,
                "model": p.model_id,
                "calibrator": p.selected,
                "raw ECE": _fmt_ci(p.metrics["raw"].get("ece")),
                "calibrated ECE": _fmt_ci(p.metrics["calibrated"].get("ece")),
                "improved": "yes"
                if raw_e and cal_e and cal_e[0] < raw_e[0]
                else "raw already better",
                "accuracy raw → cal": (
                    f"{p.metrics['raw']['accuracy']['value']:.1%} → "
                    f"{p.metrics['calibrated']['accuracy']['value']:.1%}"
                ),
                "auto-handled at ≥0.95, raw → cal": (
                    f"{_pct(trust.get('raw', {}).get('trust', {}).get('coverage'))} → "
                    f"{_pct(trust.get('calibrated', {}).get('trust', {}).get('coverage'))}"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    if report.not_evaluated:
        st.info(
            "Left out: " + "; ".join(f"{t} × {mm} ({why})" for t, mm, why in report.not_evaluated)
        )

# ---- Reliability ---------------------------------------------------------------------

with reliability:
    st.header(f"Reliability: {task} × {model_id}")
    st.caption(
        "Each point is a group of test cases with similar confidence. On the dashed diagonal, "
        "a model that says 80% is right 80% of the time. Below it: overconfident; above: "
        f"underconfident. Calibrator: {pair.selected}."
    )
    bins = pd.DataFrame(
        [
            {"side": label, **b}
            for side, label in (("raw", "raw"), ("calibrated", "calibrated"))
            for b in m["reliability"][side]["top_label"]
        ]
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color="#9a9a9a")
        .encode(x="x:Q", y="y:Q")
    )
    lines = (
        alt.Chart(bins)
        .mark_line(point=True)
        .encode(
            x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1]), title="stated confidence"),
            y=alt.Y("observed:Q", scale=alt.Scale(domain=[0, 1]), title="actual accuracy"),
            color=alt.Color("side:N", scale=COLORS, title=None),
            tooltip=[
                "side",
                alt.Tooltip("confidence:Q", format=".2f"),
                alt.Tooltip("observed:Q", format=".2f"),
                "count:Q",
            ],
        )
    )
    counts = (
        alt.Chart(bins)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("confidence:Q", bin=alt.Bin(extent=[0, 1], step=1 / 15), title=None),
            y=alt.Y("sum(count):Q", title="test cases"),
            color=alt.Color("side:N", scale=COLORS, legend=None),
            xOffset="side:N",
        )
        .properties(height=110)
    )
    st.altair_chart(
        alt.vconcat((diagonal + lines).properties(height=360), counts),
        width="stretch",
    )

# ---- Decision policy -------------------------------------------------------------------

with decision:
    st.header("Decision policy: auto-handle confident cases, send the rest to people")
    rows_df = _rows(str(ROOT), mode, task, model_id, stamp)
    threshold: float = st.slider(
        "Auto-handle when the model's confidence is at least",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.01,
        key="threshold",
    )
    left, right = st.columns(2)
    for col, side, label in (
        (left, "raw", "Raw model"),
        (right, "cal", f"Calibrated ({pair.selected})"),
    ):
        conf = rows_df[f"{side}_conf"].to_numpy(dtype=np.float64)
        correct = rows_df[f"{side}_correct"].to_numpy(dtype=bool)
        pt = data.policy(conf, correct, threshold)
        with col:
            st.subheader(label)
            st.metric(f"Auto-handled ({label})", _pct(pt.auto_share))
            st.metric(f"Error rate among auto-handled ({label})", _pct(pt.auto_error_rate))
            st.metric(f"Sent to human review ({label})", f"{pt.review_count} of {pt.n}")
    grid = [round(t, 2) for t in np.arange(0.5, 1.0001, 0.01)]
    curve: list[dict[str, Any]] = []
    for side, label in (("raw", "raw"), ("cal", "calibrated")):
        conf = rows_df[f"{side}_conf"].to_numpy(dtype=np.float64)
        correct = rows_df[f"{side}_correct"].to_numpy(dtype=bool)
        for p in data.policy_curve(conf, correct, grid):
            curve.append(
                {
                    "side": label,
                    "threshold": p.threshold,
                    "auto-handled": p.auto_share,
                    "error rate": p.auto_error_rate,
                }
            )
    curve_df = pd.DataFrame(curve).melt(
        id_vars=["side", "threshold"], var_name="measure", value_name="share"
    )
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="#666")
        .encode(x="threshold:Q")
    )
    chart = (
        alt.Chart(curve_df)
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title="confidence threshold"),
            y=alt.Y("share:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("side:N", scale=COLORS, title=None),
            strokeDash=alt.StrokeDash("measure:N", title=None),
        )
    )
    st.altair_chart((chart + rule).properties(height=320), width="stretch")
    st.caption(
        "If the raw model's confidence is dishonest, a threshold like 0.95 either auto-handles "
        "too little (under-confident) or makes more mistakes than it promises (over-confident). "
        "After calibration, the threshold means what it says."
    )

# ---- Examples ----------------------------------------------------------------------------

with examples:
    st.header(f"Test cases: {task}")
    rows_df = _rows(str(ROOT), mode, task, model_id, stamp)
    c1, c2 = st.columns(2)
    only_wrong = c1.checkbox("Only cases the calibrated model gets wrong", key="only_wrong")
    only_changed = c2.checkbox(
        "Only cases where calibration changed the decision", key="only_changed"
    )
    view = rows_df
    if only_wrong:
        view = view[~view["cal_correct"]]
    if only_changed:
        view = view[view["raw_choice"] != view["cal_choice"]]
    shown = pd.DataFrame(
        {
            "id": view["id"],
            "true label": view["label"],
            "raw": [
                f"{c} ({p:.0%})" for c, p in zip(view["raw_choice"], view["raw_conf"], strict=True)
            ],
            "calibrated": [
                f"{c} ({p:.0%})" for c, p in zip(view["cal_choice"], view["cal_conf"], strict=True)
            ],
            "case": view["text"].str.slice(0, 160),
        }
    )
    st.caption(f"{len(shown)} of {len(rows_df)} test cases · {model_id}")
    st.dataframe(shown, hide_index=True)  # pyright: ignore[reportUnknownMemberType]

    if len(view):
        case_id = st.selectbox("Look at one case", list(view["id"]), key="case") or ""
        case = rows_df[rows_df["id"] == case_id].iloc[0]
        st.markdown(f"**True label:** {case['label']}")
        st.text(str(case["text"]) or "(case text not available)")
        per_model: list[dict[str, Any]] = []
        for other in task_models:
            r = _rows(str(ROOT), mode, task, other, stamp)
            hit = r[r["id"] == case_id]
            if hit.empty:
                continue
            h = hit.iloc[0]
            per_model.append(
                {
                    "model": other,
                    "raw": f"{h['raw_choice']} ({h['raw_conf']:.0%})",
                    "calibrated": f"{h['cal_choice']} ({h['cal_conf']:.0%})",
                    "correct after calibration": "yes" if h["cal_correct"] else "no",
                }
            )
        st.dataframe(pd.DataFrame(per_model), hide_index=True)  # pyright: ignore[reportUnknownMemberType]

    # ---- live box: one typed case, scored by the real model, then calibrated ----
    st.divider()
    st.subheader("Try a case live")
    reason = live.unavailable(ROOT, task)
    if reason:
        st.info(reason)
    else:
        if "live_budget" not in st.session_state:
            st.session_state["live_budget"] = Budget(cap_usd=live.LIVE_BOX_CAP_USD)
        budget: Budget = st.session_state["live_budget"]
        st.caption(
            f"Sends the text below to {live.LIVE_MODEL} (a real, paid call; the true label is "
            f"never sent) and applies the live run's calibrator. Session cap "
            f"${live.LIVE_BOX_CAP_USD:.2f}, spent so far ${budget.spent_usd:.4f}."
        )
        with st.form("live_form"):
            fields = live.state_fields(task)
            state = {f: st.text_area(f, key=f"live_{f}") for f in fields}
            submitted = st.form_submit_button("Score this case", key="live_submit")
        if submitted:
            if not any(v.strip() for v in state.values()):
                st.warning("Type a case first.")
            else:
                try:
                    with st.spinner("Asking the model..."):
                        res = live.score_live(ROOT, task, state, budget)
                except (live.LiveBoxError, BudgetExceeded) as err:
                    st.error(str(err))
                else:
                    th = float(st.session_state.get("threshold", 0.95))
                    out: list[dict[str, Any]] = []
                    for label, probs in (
                        ("raw", res.raw),
                        (f"calibrated ({res.selected})", res.calibrated),
                    ):
                        top = max(probs, key=lambda k: probs[k])
                        out.append(
                            {
                                "": label,
                                **{k: f"{v:.0%}" for k, v in probs.items()},
                                f"decision at {th:.2f}": (
                                    f"auto: {top}" if probs[top] >= th else "send to review"
                                ),
                            }
                        )
                    st.dataframe(pd.DataFrame(out), hide_index=True)  # pyright: ignore[reportUnknownMemberType]
                    st.caption(
                        f"{res.model_id} · cost ${res.cost_usd:.5f}"
                        + (" (cached answer, free)" if res.cost_usd == 0 else "")
                    )
