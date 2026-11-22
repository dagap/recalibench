"""ApiModel through the Jev adapter against respx-mocked HTTP (no live calls)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from recalibench.core.paths import Paths
from recalibench.core.types import Item, Option, Prediction, Question
from recalibench.models.base_api import FatalApiError
from recalibench.models.budget import Budget, BudgetExceeded
from recalibench.models.context import RunContext
from recalibench.models.jev import JevModel

URL = "https://api.typesafe.ai/v1/systemone"
BINARY = Question(kind="binary", text="Did this incident result in an amputation?")
CHOICE = Question(
    kind="choice",
    text="Which event?",
    options=(Option("fall", "Falls", "A fall"), Option("contact", "Contact", "Struck by")),
)


def _items(n: int, label: int | str = 1) -> list[Item]:
    return [Item(id=f"i{i}", state={"narrative": f"case {i}"}, label=label) for i in range(n)]


def _jev(**over: Any) -> JevModel:
    kwargs: dict[str, Any] = {
        "name": "jev",
        "version": "jev-1.13.0",
        "base_url": "https://api.typesafe.ai/v1",
        "api_key_env": "TYPESAFE_API_KEY",
        "repeats": 1,
        "concurrency": 4,
        "timeout_s": 1,
        "price_per_m_input_tokens": 0.042,
        "backoff_s": 0,
        "backoff_max_s": 0,
        **over,
    }
    return JevModel(**kwargs)


def _bind(model: JevModel, tmp_path: Path, mode: str = "live", cap: float = 5.0) -> Budget:
    budget = Budget(cap_usd=cap)
    model.bind(RunContext(paths=Paths(tmp_path, mode), budget=budget))  # type: ignore[arg-type]
    return budget


def _ok(p_yes: float = 0.8, tokens: int = 100) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "jev-1.13.0",
            "answers": {"q": {"type": "noul", "noul": p_yes}},
            "usage": {"input_tokens": tokens, "output_tokens": 3},
        },
    )


def _run(model: JevModel, items: list[Item], q: Question = BINARY) -> list[Prediction]:
    return asyncio.run(model.predict(items, q))


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")


@respx.mock
def test_request_shape_and_auth(tmp_path: Path) -> None:
    route = respx.post(URL).mock(return_value=_ok())
    model = _jev()
    _bind(model, tmp_path)
    [pred] = _run(model, _items(1))
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer sk-test"
    assert json.loads(req.content) == {
        "model": "jev-1.13.0",
        "state": {"narrative": "case 0"},
        "questions": {"q": {"type": "noul", "instructions": BINARY.text}},
    }
    assert pred.probs == pytest.approx({"yes": 0.8, "no": 0.2})
    assert pred.model == "jev@jev-1.13.0" and pred.std is None
    assert pred.cost_usd == pytest.approx(100 * 0.042 / 1e6)


@respx.mock
def test_choice_roundtrip_maps_labels_to_keys(tmp_path: Path) -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "q": {"type": "choice", "probabilities": {"Falls": 0.7, "Contact": 0.3}}
                },
                "usage": {"input_tokens": 50},
            },
        )
    )
    model = _jev()
    _bind(model, tmp_path)
    [pred] = _run(model, _items(1, "fall"), CHOICE)
    criteria = json.loads(route.calls.last.request.content)["questions"]["q"]["criteria"]
    assert criteria == {"Falls": "A fall", "Contact": "Struck by"}
    assert pred.probs == pytest.approx({"fall": 0.7, "contact": 0.3})


@respx.mock
def test_retries_429_529_503_then_succeeds(tmp_path: Path) -> None:
    route = respx.post(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(529), httpx.Response(503), _ok()]
    )
    model = _jev()
    _bind(model, tmp_path)
    assert len(_run(model, _items(1))) == 1
    assert route.call_count == 4


@respx.mock
def test_retry_after_is_honoured(tmp_path: Path) -> None:
    respx.post(URL).mock(side_effect=[httpx.Response(429, headers={"Retry-After": "7"}), _ok()])
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    model = _jev(backoff_max_s=60)
    model._sleep = fake_sleep  # pyright: ignore[reportPrivateUsage]
    _bind(model, tmp_path)
    _run(model, _items(1))
    assert slept == [7.0]


@respx.mock
@pytest.mark.parametrize("status", [400, 401, 422])
def test_client_errors_are_fatal_without_retry(tmp_path: Path, status: int) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(status, json={"error": "nope"}))
    model = _jev()
    budget = _bind(model, tmp_path)
    with pytest.raises(FatalApiError, match=f"HTTP {status}"):
        _run(model, _items(3))
    assert route.call_count <= 3  # each in-flight item tried once, none retried
    assert budget.reserved_usd == pytest.approx(0) and budget.spent_usd == 0


@respx.mock
def test_exhausted_retries_drop_only_that_item(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        return httpx.Response(503) if state["narrative"] == "case 1" else _ok()

    route = respx.post(URL).mock(side_effect=respond)
    model = _jev(max_attempts=5)
    budget = _bind(model, tmp_path)
    preds = _run(model, _items(3))
    assert [p.item_id for p in preds] == ["i0", "i2"]
    assert "gave up after 5 attempts" in model.last_errors["i1"]
    assert route.call_count == 2 + 5
    assert budget.reserved_usd == pytest.approx(0)


@respx.mock
def test_timeouts_are_retried_then_dropped(tmp_path: Path) -> None:
    respx.post(URL).mock(side_effect=[httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), _ok()])
    model = _jev()
    _bind(model, tmp_path)
    assert len(_run(model, _items(1))) == 1

    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    model = _jev(max_attempts=3)
    _bind(model, tmp_path / "other")
    assert _run(model, _items(1)) == []
    assert "timeout" in model.last_errors["i0"]


@respx.mock
def test_unparseable_reply_drops_item_and_is_billed(tmp_path: Path) -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(200, json={"model": "jev-1.14.0", "answers": {}})
    )
    model = _jev()
    budget = _bind(model, tmp_path)
    assert _run(model, _items(1)) == []
    assert "pinned 'jev-1.13.0'" in model.last_errors["i0"]
    assert budget.spent_usd > 0  # the estimate is charged: the call was made


@respx.mock
def test_second_run_is_served_from_cache(tmp_path: Path) -> None:
    route = respx.post(URL).mock(return_value=_ok())
    model = _jev(repeats=2)
    budget = _bind(model, tmp_path)
    first = _run(model, _items(5))
    spent = budget.spent_usd
    assert route.call_count == 10
    second = _run(model, _items(5))
    assert route.call_count == 10  # zero new calls
    assert budget.spent_usd == spent  # re-runs never re-bill
    assert [p.probs for p in first] == [p.probs for p in second]
    assert (tmp_path / "cache" / "jev@jev-1.13.0").is_dir()

    other = Question(kind="binary", text="Was anyone hospitalised?")
    _run(model, _items(5), other)
    assert route.call_count == 20  # a different question misses the cache


@respx.mock
def test_budget_estimate_refuses_before_any_call(tmp_path: Path) -> None:
    route = respx.post(URL).mock(return_value=_ok())
    model = _jev(repeats=3)
    budget = _bind(model, tmp_path, cap=1e-9)
    with pytest.raises(BudgetExceeded, match="exceeds the budget"):
        budget.check_estimate(model.estimate_cost(_items(10), BINARY))
    assert route.call_count == 0


@respx.mock
def test_budget_stops_the_run_at_the_cap(tmp_path: Path) -> None:
    model = _jev(concurrency=1)
    tokens = model.count_tokens(model.build_request(_items(1)[0], {"q": BINARY}))
    route = respx.post(URL).mock(return_value=_ok(tokens=tokens))  # actual == estimate
    per_call = model.estimate_cost(_items(1), BINARY)
    budget = _bind(model, tmp_path, cap=per_call * 3.5)
    with pytest.raises(BudgetExceeded, match="cap"):
        _run(model, _items(50))
    assert budget.spent_usd <= budget.cap_usd
    assert 3 <= route.call_count < 50


@respx.mock
def test_one_budget_is_shared_by_two_models(tmp_path: Path) -> None:
    respx.post(URL).mock(return_value=_ok(tokens=10_000_000))  # $0.42 per call
    budget = Budget(cap_usd=1.0)
    a, b = _jev(name="jev_a", concurrency=1), _jev(name="jev_b", concurrency=1)
    for m in (a, b):
        m.count_tokens = lambda request: 10_000_000  # type: ignore[method-assign]
        m.bind(RunContext(paths=Paths(tmp_path, "live"), budget=budget))
    _run(a, _items(2))  # $0.84
    with pytest.raises(BudgetExceeded):
        _run(b, _items(2))  # the second model can't spend the first one's share
    assert budget.spent_usd <= 1.0
    assert set(budget.spent_by_model) == {"jev_a@jev-1.13.0"}


@respx.mock
def test_concurrency_is_bounded(tmp_path: Path) -> None:
    in_flight = peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _ok()

    respx.post(URL).mock(side_effect=respond)
    model = _jev(concurrency=2)
    _bind(model, tmp_path)
    assert len(_run(model, _items(12))) == 12
    assert peak == 2


@respx.mock
def test_repeats_are_averaged_with_std(tmp_path: Path) -> None:
    route = respx.post(URL).mock(side_effect=[_ok(0.6), _ok(0.8), _ok(1.0)])
    model = _jev(repeats=3, concurrency=1)
    _bind(model, tmp_path)
    [pred] = _run(model, _items(1))
    assert route.call_count == 3
    assert pred.probs["yes"] == pytest.approx(0.8)
    assert pred.std is not None and pred.std["yes"] == pytest.approx((0.08 / 3) ** 0.5)


@respx.mock(assert_all_called=False)
def test_mock_mode_runs_the_adapter_offline(tmp_path: Path) -> None:
    route = respx.post(URL)
    model = _jev(repeats=3)
    budget = _bind(model, tmp_path, mode="mock")
    calls: list[str] = []
    real_build, real_parse = model.build_request, model.parse_response
    model.build_request = lambda i, q: calls.append("build") or real_build(i, q)  # type: ignore[method-assign]
    model.parse_response = lambda r, q: calls.append("parse") or real_parse(r, q)  # type: ignore[method-assign]
    preds = _run(model, _items(4), BINARY)
    preds += _run(model, _items(3, "contact"), CHOICE)
    assert route.call_count == 0 and budget.spent_usd == 0
    assert len(preds) == 7 and all(p.cost_usd == 0 for p in preds)
    assert calls.count("build") == calls.count("parse") == 7 * 3
    assert (tmp_path / "cache_mock" / "jev@jev-1.13.0").is_dir()
    assert not (tmp_path / "cache").exists()


def test_predict_requires_bind() -> None:
    with pytest.raises(RuntimeError, match="bind"):
        _run(_jev(), _items(1))


# ---- batching (off by default; see base_api.py) ------------------------------------

PHRASINGS = [
    Question(kind="binary", text="Did this incident result in an amputation?"),
    Question(kind="binary", text="Was a body part lost in this incident?"),
    Question(kind="binary", text="Did the worker suffer an amputation?"),
]


def _echo(request: httpx.Request) -> httpx.Response:
    """Answer every named question; P(yes) encodes which phrasing was asked."""
    body = json.loads(request.content)
    answers = {
        name: {
            "type": "noul",
            "noul": 0.1 * (1 + [q.text for q in PHRASINGS].index(q["instructions"])),
        }
        for name, q in body["questions"].items()
    }
    return httpx.Response(
        200,
        json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 90}},
    )


@pytest.mark.parametrize(
    ("batch_questions", "batch_repeats", "calls_per_item", "questions_per_call"),
    [(False, False, 6, 1), (True, False, 2, 3), (False, True, 3, 2), (True, True, 1, 6)],
)
@respx.mock
def test_batching_groups_requests(
    tmp_path: Path,
    batch_questions: bool,
    batch_repeats: bool,
    calls_per_item: int,
    questions_per_call: int,
) -> None:
    route = respx.post(URL).mock(side_effect=_echo)
    model = _jev(repeats=2, batch_questions=batch_questions, batch_repeats=batch_repeats)
    budget = _bind(model, tmp_path)
    per_q = asyncio.run(model.predict_questions(_items(4), PHRASINGS))
    assert route.call_count == 4 * calls_per_item
    requests: list[httpx.Request] = [call.request for call in route.calls]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    sizes = {len(json.loads(r.content)["questions"]) for r in requests}
    assert sizes == {questions_per_call}
    for qi, preds in enumerate(per_q):  # each phrasing got its own answers back
        assert [p.probs["yes"] for p in preds] == pytest.approx([0.1 * (qi + 1)] * 4)
    # Actual spend: 90 tokens per request, however many questions it carried.
    assert budget.spent_usd == pytest.approx(4 * calls_per_item * 90 * 0.042 / 1e6)
    per_pred = sum(p.cost_usd for preds in per_q for p in preds)
    assert per_pred == pytest.approx(budget.spent_usd)  # cost is shared out, not double-counted


@respx.mock
def test_batched_answers_are_cached_apart_from_unbatched(tmp_path: Path) -> None:
    route = respx.post(URL).mock(side_effect=_echo)
    plain = _jev()
    _bind(plain, tmp_path)
    asyncio.run(plain.predict_questions(_items(2), PHRASINGS))
    assert route.call_count == 6
    batched = _jev(batch_questions=True)
    _bind(batched, tmp_path)
    asyncio.run(batched.predict_questions(_items(2), PHRASINGS))
    assert route.call_count == 8  # batched answers may differ, so no reuse
    asyncio.run(batched.predict_questions(_items(2), PHRASINGS))
    assert route.call_count == 8  # but batched re-runs hit their own cache


@respx.mock
def test_batched_failure_drops_the_item_for_every_question_it_carried(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["state"]["narrative"] == "case 1":
            return httpx.Response(503)
        return _echo(request)

    respx.post(URL).mock(side_effect=respond)
    model = _jev(batch_questions=True, max_attempts=2)
    _bind(model, tmp_path)
    per_q = asyncio.run(model.predict_questions(_items(3), PHRASINGS))
    assert all([p.item_id for p in preds] == ["i0", "i2"] for preds in per_q)
    assert "i1" in model.last_errors


@respx.mock
def test_budget_reserves_the_unbatched_worst_case(tmp_path: Path) -> None:
    route = respx.post(URL).mock(side_effect=_echo)
    model = _jev(batch_questions=True)
    single = model.estimate_cost(_items(1), PHRASINGS[0])
    _bind(model, tmp_path, cap=single * 2.5)  # room for 2 single questions, not 3
    with pytest.raises(BudgetExceeded):
        asyncio.run(model.predict_questions(_items(1), PHRASINGS))
    assert route.call_count == 0
    singles = sum(model.estimate_cost(_items(1), q) for q in PHRASINGS)
    assert model.estimate_cost_questions(_items(1), PHRASINGS) == pytest.approx(singles)


@respx.mock(assert_all_called=False)
def test_batching_works_in_mock_mode(tmp_path: Path) -> None:
    route = respx.post(URL)
    model = _jev(repeats=3, batch_questions=True, batch_repeats=True)
    _bind(model, tmp_path, mode="mock")
    per_q = asyncio.run(model.predict_questions(_items(5), PHRASINGS))
    assert route.call_count == 0
    assert [len(p) for p in per_q] == [5, 5, 5]
    assert all(p.std is not None and p.std["yes"] > 0 for preds in per_q for p in preds)


@respx.mock
def test_latency_excludes_time_queued_for_a_concurrency_slot(tmp_path: Path) -> None:
    # Live run 2026-11-22 reported ~31 s per item: the timer started before the request
    # waited for one of `concurrency` slots. Latency must be the call itself.
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return _ok()

    respx.post(URL).mock(side_effect=slow)
    model = _jev(concurrency=1, repeats=1)
    _bind(model, tmp_path)
    preds = _run(model, _items(8))  # 8 calls in single file: ~0.4 s of queueing for the last
    assert len(preds) == 8
    latencies = [p.latency_ms for p in preds]
    assert all(x is not None and x < 200 for x in latencies), latencies
