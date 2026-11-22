"""Score stage end to end on the toy task, with Jev against respx in live mode."""

import json
import shutil
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pandas as pd
import pytest
import respx

from recalibench.core.config import DEFAULT_CONF_DIR, load_config
from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.core.types import Item, Prediction, Question
from recalibench.models.budget import BudgetExceeded
from recalibench.pipeline import download, prepare
from recalibench.pipeline.score import PairResult, score
from recalibench.pipeline.scores_io import read_scores
from tests.conftest import TOY_CONF

URL = "https://api.typesafe.ai/v1/systemone"


class SpyModel:
    """Trainable stand-in that records what each instance trained on and scored."""

    supports = frozenset({"binary", "choice"})
    trainable = True
    log: ClassVar[list[tuple[frozenset[str], list[str]]]] = []

    def __init__(self, name: str, version: str = "1") -> None:
        self.name, self.version = name, version
        self.trained: frozenset[str] = frozenset()

    def fit(self, items: list[Item], question: Question) -> None:
        self.trained = frozenset(it.id for it in items)

    async def predict(self, items: list[Item], question: Question) -> list[Prediction]:
        SpyModel.log.append((self.trained, [it.id for it in items]))
        keys = question.keys
        return [
            Prediction(it.id, {k: 1 / len(keys) for k in keys}, None, f"{self.name}@1", None, 0.0)
            for it in items
        ]

    def estimate_cost(self, items: list[Item], question: Question) -> float:
        return 0.0


@pytest.fixture
def conf(tmp_path: Path) -> Path:
    """Extra configs: the real jev config made fast (1 repeat, no retries), and the spy."""
    d = tmp_path / "conf"
    (d / "model").mkdir(parents=True)
    jev = (DEFAULT_CONF_DIR / "model" / "jev.yaml").read_text()
    jev = jev.replace("repeats: 3", "repeats: 1").replace("max_attempts: 5", "max_attempts: 1")
    (d / "model" / "jev_fast.yaml").write_text(jev)
    (d / "model" / "spy.yaml").write_text("_target_: tests.integration.test_score.SpyModel\n")
    return d


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")


def _setup(tmp_path: Path, conf: Path, *overrides: str) -> tuple[Registry, Paths, Any]:
    dirs = [TOY_CONF, conf]
    ov = [f"paths.root={tmp_path}", "experiment.tasks=[toy_binary]", *overrides]
    cfg = load_config(ov, extra_conf_dirs=dirs)
    reg = Registry(extra_conf_dirs=dirs, overrides=ov)
    paths = Paths.from_config(cfg)
    if not paths.processed("toy_binary").exists():
        prep_cfg = load_config(
            [f"paths.root={tmp_path}", "experiment.tasks=[toy_binary,toy_choice]"], dirs
        )
        download.download(reg, paths, ["toy"])
        prepare.prepare(reg, Paths.from_config(prep_cfg), prep_cfg)
    return reg, paths, cfg


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "jev-1.14.0",
            "answers": {"q": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 40},
        },
    )


def _status(results: list[PairResult]) -> dict[str, str]:
    return {f"{r.task}/{r.model.split('@')[0]}": r.status for r in results}


@respx.mock
def test_second_run_makes_zero_network_calls(tmp_path: Path, conf: Path) -> None:
    route = respx.post(URL).mock(side_effect=_ok)
    reg, paths, cfg = _setup(tmp_path, conf, "mode=live", "experiment.models=[jev_fast]")
    assert _status(score(reg, paths, cfg)) == {"toy_binary/jev_fast": "scored"}
    assert route.call_count == 200

    assert _status(score(reg, paths, cfg)) == {"toy_binary/jev_fast": "fresh"}
    assert route.call_count == 200  # skipped by fingerprint

    forced = score(reg, paths, cfg, force=True)
    assert route.call_count == 200  # re-scored, every answer from the response cache
    assert forced[0].estimate_usd == 0.0 and forced[0].cost_usd == 0.0

    shutil.rmtree(tmp_path / "cache")
    score(reg, paths, cfg, force=True)
    assert route.call_count == 400  # no cache, so the calls come back

    s = read_scores(paths.artifact_dir("toy_binary", "jev_fast", "jev-1.14.0"))
    assert len(s.frame) == 200 and set(s.frame["split"]) == {"fit", "val", "test"}
    assert s.meta["errors"] == 0 and s.meta["model_id"] == "jev_fast@jev-1.14.0"


@respx.mock
@pytest.mark.parametrize(("failing", "status"), [(2, "scored"), (6, "failed")])
def test_error_threshold(tmp_path: Path, conf: Path, failing: int, status: str) -> None:
    reg, paths, cfg = _setup(tmp_path, conf, "mode=live", "experiment.models=[jev_fast]")
    texts = [
        json.loads(s)["text"] for s in pd.read_parquet(paths.processed("toy_binary"))["state_json"]
    ]
    bad = set(texts[:failing])

    def respond(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["state"]["text"] in bad:
            return httpx.Response(503)
        return _ok(request)

    respx.post(URL).mock(side_effect=respond)
    [result] = score(reg, paths, cfg)
    assert result.status == status and result.errors == failing
    out = paths.artifact_dir("toy_binary", "jev_fast", "jev-1.14.0")
    if status == "scored":
        assert read_scores(out).meta["errors"] == failing
        assert "gave up" in next(iter(read_scores(out).meta["error_samples"].values()))
    else:
        assert "items failed" in result.reason and not (out / "scores.parquet").exists()


@respx.mock(assert_all_called=False)
def test_dry_run_estimates_without_calling(tmp_path: Path, conf: Path) -> None:
    route = respx.post(URL)
    reg, paths, cfg = _setup(
        tmp_path, conf, "mode=live", "experiment.models=[jev_fast,tfidf_logreg]", "dry_run=true"
    )
    results = score(reg, paths, cfg)
    assert {r.status for r in results} == {"estimated"} and route.call_count == 0
    jev = next(r for r in results if r.model.startswith("jev_fast"))
    assert jev.estimate_usd > 0
    assert not (tmp_path / "artifacts").exists()


@respx.mock(assert_all_called=False)
def test_live_budget_refuses_before_any_call(tmp_path: Path, conf: Path) -> None:
    route = respx.post(URL)
    reg, paths, cfg = _setup(
        tmp_path, conf, "mode=live", "experiment.models=[jev_fast]", "budget_usd=0.0000001"
    )
    with pytest.raises(BudgetExceeded, match="exceeds the budget"):
        score(reg, paths, cfg)
    assert route.call_count == 0


@respx.mock
def test_limit_smoke_run_fills_cache_but_writes_nothing(tmp_path: Path, conf: Path) -> None:
    route = respx.post(URL).mock(side_effect=_ok)
    reg, paths, cfg = _setup(
        tmp_path, conf, "mode=live", "experiment.models=[jev_fast]", "limit=20"
    )
    [r] = score(reg, paths, cfg)
    assert r.status == "smoke" and 20 <= route.call_count <= 22
    assert not paths.artifact_dir("toy_binary", "jev_fast", "jev-1.14.0").exists()
    full_cfg = load_config(
        [
            f"paths.root={tmp_path}",
            "experiment.tasks=[toy_binary]",
            "mode=live",
            "experiment.models=[jev_fast]",
        ],
        [TOY_CONF, conf],
    )
    score(reg, paths, full_cfg)
    assert route.call_count == 200  # the full run reuses the smoke run's answers


def test_trainable_scores_are_out_of_fold(tmp_path: Path, conf: Path) -> None:
    SpyModel.log.clear()
    reg, paths, cfg = _setup(tmp_path, conf, "experiment.models=[spy]")
    [r] = score(reg, paths, cfg)
    assert r.status == "scored"
    s = read_scores(paths.artifact_dir("toy_binary", "spy", "1"))
    split = dict(zip(s.frame["id"], s.frame["split"], strict=True))
    fit_ids = frozenset(i for i, sp in split.items() if sp == "fit")
    for trained, predicted in SpyModel.log:
        for item_id in predicted:
            assert item_id not in trained  # nobody scores an item it trained on
            if split[item_id] != "fit":
                assert trained == fit_ids  # val/test: the model fitted on the whole fit split
    oof = [p for t, p in SpyModel.log if t != fit_ids]
    assert len(oof) == 5 and sorted(i for p in oof for i in p) == sorted(fit_ids)


def test_unsupported_pairs_are_skipped_with_a_reason(tmp_path: Path, conf: Path) -> None:
    reg, paths, cfg = _setup(
        tmp_path,
        conf,
        "experiment.tasks=[toy_choice]",
        "experiment.models=[toy,structured_logreg,tfidf_logreg]",
    )
    results = {r.model.split("@")[0]: r for r in score(reg, paths, cfg)}
    assert results["toy"].status == "skipped" and "supports ['binary']" in results["toy"].reason
    assert results["structured_logreg"].status == "skipped"
    assert "no non-text fields" in results["structured_logreg"].reason
    assert results["tfidf_logreg"].status == "scored"
    assert not (tmp_path / "artifacts_mock" / "toy_choice" / "toy@0.1").exists()
