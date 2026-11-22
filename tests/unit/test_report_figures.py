from pathlib import Path

import pytest

from recalibench.core.paths import Paths
from recalibench.report import figures
from recalibench.report.collect import Pair, collect
from tests.unit.report_fixtures import option_order_summary, wording_summary, write_metrics


def _pairs(tmp_path: Path, **kwargs: object) -> list[Pair]:
    arts = tmp_path / "artifacts_mock"
    write_metrics(arts, task="t", model_id="bin@1", kind="binary", **kwargs)  # type: ignore[arg-type]
    write_metrics(arts, task="t", model_id="cho@1", kind="choice")
    return collect(Paths(tmp_path, "mock")).pairs


def _has_series(svg: str) -> None:
    assert svg.startswith("<svg")
    assert 'id="raw"' in svg and 'id="calibrated"' in svg


def test_every_pair_figure_has_data(tmp_path: Path) -> None:
    for pair in _pairs(tmp_path):
        for fn in (figures.reliability_svg, figures.confidence_hist_svg, figures.risk_coverage_svg):
            _has_series(fn(pair))


def test_reliability_shows_bin_counts_and_diagonal(tmp_path: Path) -> None:
    svg = figures.reliability_svg(_pairs(tmp_path)[0])
    assert 'id="counts"' in svg and 'id="diagonal"' in svg


def test_error_lengths_clip_when_ci_excludes_point() -> None:
    assert figures.error_lengths(0.0163, 0.0185, 0.0518) == (0.0, pytest.approx(0.0355))
    assert figures.error_lengths(0.1, None, None) == (0.0, 0.0)


def test_cross_model_chart_with_ci_below_point(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path, cal_ece=0.0163, cal_ece_ci=(0.0185, 0.0518))
    _has_series(figures.cross_model_ece_svg("t", pairs))


def test_experiment_figures() -> None:
    _has_series(figures.wording_svg(wording_summary()))
    _has_series(figures.option_order_svg(option_order_summary()))


def test_svg_is_deterministic(tmp_path: Path) -> None:
    pair = _pairs(tmp_path)[0]
    assert figures.reliability_svg(pair) == figures.reliability_svg(pair)
