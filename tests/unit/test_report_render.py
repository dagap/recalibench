import re
from pathlib import Path
from typing import Any

from recalibench.core.paths import Paths
from recalibench.report.collect import ReportData, collect
from recalibench.report.render import render
from tests.unit.report_fixtures import (
    option_order_summary,
    transfer_summary,
    wording_summary,
    write_metrics,
    write_summary,
)


def _data(tmp_path: Path, experiments: dict[str, dict[str, Any]] | None = None) -> ReportData:
    arts = tmp_path / "artifacts_mock"
    write_metrics(
        arts, task="t_bin", model_id="m@1", kind="binary", cost_usd=0.5, mean_latency_ms=120.0
    )
    write_metrics(arts, task="t_bin", model_id="n@1", kind="binary", raw_ece=0.02, cal_ece=0.03)
    write_metrics(arts, task="t_cho", model_id="m@1", kind="choice", cost_usd=0.25)
    for name, s in (experiments or {}).items():
        write_summary(arts, name, s)
    return collect(Paths(tmp_path, "mock"))


def _render(data: ReportData) -> str:
    return render(data, "mode: mock\nseed: 42\n", "2026-11-22T00:00:00+00:00")


def _section(html: str, sid: str) -> str:
    start = html.index(f'id="{sid}"')
    end = html.find("<section", start)
    return html[start : end if end != -1 else len(html)]


def test_sections_present(tmp_path: Path) -> None:
    exps = {
        "wording": wording_summary(),
        "option_order": option_order_summary(),
        "transfer": transfer_summary(),
    }
    html = _render(_data(tmp_path, exps))
    assert html.startswith("<!doctype html>")
    for sid in (
        "summary",
        "task-t_bin",
        "task-t_cho",
        "pair-t_bin-m@1",
        "pair-t_bin-n@1",
        "pair-t_cho-m@1",
        "experiments",
        "cost",
        "config",
    ):
        assert f'id="{sid}"' in html, sid
    assert html.count("<svg") == 3 * 3 + 2 + 2  # pairs x 3, per-task charts, 2 experiment charts
    assert "seed: 42" in _section(html, "config")
    assert "MOCK" in html


def test_self_contained(tmp_path: Path) -> None:
    html = _render(_data(tmp_path, {"wording": wording_summary()}))
    for bad in ('src="http', 'href="http', "<link", "<script src", "url(http"):
        assert bad not in html


def test_raw_already_better_is_flagged(tmp_path: Path) -> None:
    summary = _section(_render(_data(tmp_path)), "summary")
    rows = {
        re.sub(r"<[^>]+>", "", r.split("</th>")[0].split(">", 1)[-1]): r
        for r in summary.split("<tr")[1:]
    }
    assert "raw already better" in rows["t_bin"]  # n@1: 0.02 -> 0.03
    assert summary.count("raw already better") == 1  # m@1: 0.10 -> 0.02 is not flagged


def test_choice_pair_shows_dash_for_auroc(tmp_path: Path) -> None:
    pair = _section(_render(_data(tmp_path)), "pair-t_cho-m@1")
    auroc_row = re.search(r"<tr><th[^>]*>auroc</th>(.*?)</tr>", pair)
    assert auroc_row is not None and "—" in auroc_row.group(1)


def test_untrusted_text_is_escaped(tmp_path: Path) -> None:
    w = wording_summary()
    w["variants"][0]["question"] = 'Is <b>x</b> & "y"?'
    html = _render(_data(tmp_path, {"wording": w}))
    assert "Is &lt;b&gt;x&lt;/b&gt; &amp; &quot;y&quot;?" in html
    assert "<b>x</b>" not in html


def test_unknown_or_partial_experiment_falls_back(tmp_path: Path) -> None:
    html = _render(_data(tmp_path, {"mystery": {"a": 1}, "wording": {"variants": []}}))
    exps = _section(html, "experiments")
    assert "mystery" in exps and "wording" in exps


def test_not_evaluated_listed(tmp_path: Path) -> None:
    data = _data(tmp_path)
    data.not_evaluated.append(("t_bin", "late@9"))
    assert "late@9" in _section(_render(data), "summary")


def test_cost_per_model(tmp_path: Path) -> None:
    cost = _section(_render(_data(tmp_path)), "cost")
    assert "$0.7500" in cost  # m@1: 0.5 + 0.25 across its two tasks
    assert "120" in cost  # m@1 mean latency over pairs that report one
