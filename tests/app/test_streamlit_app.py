"""The Streamlit app on toy mock artifacts, through streamlit's own AppTest harness."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from recalibench.core.config import load_config
from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.pipeline import calibrate, download, evaluate, prepare, score
from tests.conftest import TOY_CONF

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("toy_app")
    ov = [
        f"paths.root={root}",
        "experiment.tasks=[toy_binary,toy_choice]",
        "experiment.models=[mock,tfidf_logreg]",
        "bootstrap_iters=50",
    ]
    cfg = load_config(ov, extra_conf_dirs=[TOY_CONF])
    reg = Registry(extra_conf_dirs=[TOY_CONF], overrides=ov)
    paths = Paths.from_config(cfg)
    download.download(reg, paths, ["toy"])
    prepare.prepare(reg, paths, cfg)
    score.score(reg, paths, cfg)
    calibrate.calibrate(reg, paths, cfg)
    evaluate.evaluate(reg, paths, cfg)
    return root


@pytest.fixture
def app(toy_root: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("RECAL_ROOT", str(toy_root))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_every_tab_renders(app: AppTest) -> None:
    assert [t.label for t in app.tabs] == ["Overview", "Reliability", "Decision policy", "Examples"]
    assert app.get("vega_lite_chart") or app.get("arrow_vega_lite_chart")  # charts drawn
    assert len(app.dataframe) >= 2  # overview matrix + examples table


def test_pickers_list_what_is_on_disk(app: AppTest) -> None:
    assert app.selectbox(key="task").options == ["toy_binary", "toy_choice"]
    assert app.selectbox(key="model").options == ["mock@mock-1", "tfidf_logreg@tfidf-lr-1"]
    app.selectbox(key="task").set_value("toy_choice").run()
    assert not app.exception


def test_slider_changes_policy_numbers(app: AppTest) -> None:
    def shares() -> list[str]:
        return [m.value for m in app.metric if m.label.startswith("Auto-handled")]

    app.slider(key="threshold").set_value(0.99).run()
    high = shares()
    app.slider(key="threshold").set_value(0.5).run()
    low = shares()
    assert len(high) == 2 and high != low  # raw and calibrated, both move
    assert all(v == "100%" for v in low)  # at 0.5 every top-label case is auto-handled


def test_examples_filters(app: AppTest) -> None:
    app.checkbox(key="only_wrong").check().run()
    assert not app.exception
    app.checkbox(key="only_changed").check().run()
    assert not app.exception


def test_no_artifacts_shows_how_to_build_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECAL_ROOT", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("make all" in e.value for e in at.error)


def test_live_box_hidden_without_key(app: AppTest) -> None:
    assert "live_submit" not in [b.key for b in app.button]
    assert any("TYPESAFE_API_KEY" in i.value for i in app.info)


def test_live_box_needs_a_live_run(toy_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAL_ROOT", str(toy_root))
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")  # never used: no live artifacts
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert "live_submit" not in [b.key for b in at.button]
    assert any("live run" in i.value for i in at.info)
