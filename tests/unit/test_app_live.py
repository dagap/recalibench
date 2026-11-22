"""Live box: one typed case -> real Jev adapter (respx, no network) -> selected calibrator."""

import json
import pickle
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from recalibench.app import live
from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.models.budget import Budget, BudgetExceeded

URL = "https://api.typesafe.ai/v1/systemone"
TASK = "osha_amputation"  # binary, state {narrative: ...}
MODEL_ID = "jev@jev-1.14.0"


def _live_artifacts(root: Path, model_id: str = MODEL_ID) -> None:
    """A live selected.json + fitted Platt calibrator, as the calibrate stage writes them."""
    d = Paths(root, "live").artifacts_root() / TASK / MODEL_ID
    (d / "calibrators").mkdir(parents=True)
    rng = np.random.default_rng(0)
    p_yes = rng.uniform(0.3, 0.7, 400)  # under-confident raw scores
    y = (rng.uniform(size=400) < np.clip((p_yes - 0.5) * 2.5 + 0.5, 0, 1)).astype(np.int64)
    P = np.column_stack([p_yes, 1 - p_yes])
    cal = Registry().calibrator("platt")
    cal.fit(P, 1 - y)  # y is the true class's column index: yes = column 0
    (d / "calibrators" / "platt.pkl").write_bytes(pickle.dumps(cal))
    (d / "selected.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "model_id": model_id,
                "kind": "binary",
                "keys": ["yes", "no"],
                "selected": "platt",
                "scores_fingerprint": "x",
            }
        )
    )


SENT: list[dict[str, Any]] = []


def _reply(request: httpx.Request) -> httpx.Response:
    body: dict[str, Any] = json.loads(request.content)
    SENT.append(body)
    answers = {n: {"type": "noul", "noul": 0.62} for n in body["questions"]}
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "answers": answers,
            "usage": {"input_tokens": 320, "output_tokens": 0},
        },
    )


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")


def test_fields_come_from_the_task_config() -> None:
    assert live.state_fields(TASK) == ["narrative"]


def test_unavailable_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    _live_artifacts(tmp_path)
    assert "TYPESAFE_API_KEY" in (live.unavailable(tmp_path, TASK) or "")


def test_unavailable_with_mock_only_artifacts(tmp_path: Path, key: None) -> None:
    (tmp_path / "artifacts_mock" / TASK / MODEL_ID).mkdir(parents=True)
    assert "live run" in (live.unavailable(tmp_path, TASK) or "")


def test_available_with_key_and_live_artifacts(tmp_path: Path, key: None) -> None:
    _live_artifacts(tmp_path)
    assert live.unavailable(tmp_path, TASK) is None


@respx.mock
def test_scores_one_case_and_applies_the_calibrator(tmp_path: Path, key: None) -> None:
    _live_artifacts(tmp_path)
    route = respx.post(URL).mock(side_effect=_reply)
    SENT.clear()
    budget = Budget(cap_usd=live.LIVE_BOX_CAP_USD)
    res = live.score_live(tmp_path, TASK, {"narrative": "Worker lost a fingertip."}, budget)

    sent = SENT[0]
    assert sent["state"] == {"narrative": "Worker lost a fingertip."}  # the state, nothing else
    assert sent["model"] == "jev-1.14.0"
    assert res.raw["yes"] == pytest.approx(0.62)
    assert res.selected == "platt"
    assert res.calibrated["yes"] != pytest.approx(0.62)  # the calibrator moved it
    assert res.calibrated["yes"] + res.calibrated["no"] == pytest.approx(1.0)
    assert res.cost_usd > 0

    again = live.score_live(tmp_path, TASK, {"narrative": "Worker lost a fingertip."}, budget)
    assert again.raw == res.raw and again.cost_usd == 0.0  # cached: asking again is free
    assert route.call_count == jev_requests_for_one_case()


def jev_requests_for_one_case() -> int:
    # jev.yaml batches its repeats, so one case is one request
    return 1


@respx.mock
def test_budget_cap_refuses_before_any_call(tmp_path: Path, key: None) -> None:
    _live_artifacts(tmp_path)
    route = respx.post(URL).mock(side_effect=_reply)
    with pytest.raises(BudgetExceeded):
        live.score_live(tmp_path, TASK, {"narrative": "x"}, Budget(cap_usd=1e-9))
    assert route.call_count == 0


def test_calibrator_for_another_model_version_is_refused(tmp_path: Path, key: None) -> None:
    _live_artifacts(tmp_path, model_id="jev@jev-1.13.0")
    with pytest.raises(live.LiveBoxError, match="jev-1.13.0"):
        live.score_live(tmp_path, TASK, {"narrative": "x"}, Budget(cap_usd=1.0))
