"""ApiModel: the base class for hosted models.

An adapter (e.g. jev.py) implements only the wire format. A request carries one
or more named questions about one item (`questions: {name: Question}`):
    build_request(item, questions) -> dict             the JSON body for one call
    parse_response(raw, questions) -> ({name: probs}, Usage)
    fake_response(request, questions, {name: probs})  a reply shaped like the real API
    count_tokens(request) -> int                       estimate, for the budget
    price(usage) -> float                              USD for one call
    headers() -> dict                                  auth etc. (read keys lazily here)

The base supplies: bounded concurrency, `repeats` answers per item and question
averaged into probs (with their std), retries with jitter on 429/5xx/timeouts,
the response cache (one entry per item x question x repeat), the shared budget,
cost and latency tracking, per-item error isolation, and mock routing: with
mode=mock (or `always_mock`) the network call is replaced by MockGenerator +
fake_response, so build/parse code still runs offline.

Batching (both off by default; whether it saves money depends on how the
provider bills a shared state, which the Jev docs don't say):
    batch_questions  all questions of predict_questions() for an item in one request
    batch_repeats    all repeats of a question in one request (may return identical
                     answers if the provider is deterministic within a request)
The budget always reserves the unbatched (worst-case) cost and settles the
actual `usage`. Batched answers are cached apart from unbatched ones.

Items that fail (retries exhausted, unparseable reply) are logged, recorded in
`last_errors` and left out of the result for every question their request
carried. Fatal API errors (401, 400, 422, ...) and BudgetExceeded stop the call.
"""

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from recalibench.core.types import Item, Prediction, Question
from recalibench.models.budget import BudgetExceeded
from recalibench.models.cache import ResponseCache, cache_key
from recalibench.models.context import RunContext
from recalibench.models.mock_generator import MockGenerator

log = logging.getLogger(__name__)

PROB_SUM_TOLERANCE = 1e-3
SINGLE = "q"  # question name when a request carries one question


class ParseError(Exception):
    """A reply could not be turned into a valid probability per option."""


class RetryableError(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FatalApiError(Exception):
    """Every call would fail the same way (bad key, invalid request): stop the run."""


class ItemFailed(Exception):
    pass


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int = 0


@dataclass(frozen=True)
class CallResult:
    probs: dict[str, float]
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class Slot:
    """One answer we need: question `q_index`, repeat `repeat`, cached under `key`."""

    q_index: int
    repeat: int
    key: str


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return None if value is None else max(0.0, float(value))
    except ValueError:
        return None


def validate_probs(probs: Mapping[str, float], question: Question) -> dict[str, float]:
    """Exactly the question's keys, finite, in [0, 1], summing to 1 (renormalised)."""
    keys = set(question.keys)
    if set(probs) != keys:
        raise ParseError(f"expected probabilities for {sorted(keys)}, got {sorted(probs)}")
    values = list(probs.values())
    if any(not math.isfinite(v) or v < -1e-9 or v > 1 + 1e-9 for v in values):
        raise ParseError(f"probabilities out of range: {dict(probs)}")
    total = sum(values)
    if abs(total - 1.0) > PROB_SUM_TOLERANCE:
        raise ParseError(f"probabilities sum to {total:.6f}, not 1")
    return {k: min(1.0, max(0.0, v)) / total for k, v in probs.items()}


class ApiModel:
    supports: frozenset[str] = frozenset()
    trainable: bool = False
    always_mock: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        version: str,
        base_url: str,
        api_key_env: str,
        repeats: int = 1,
        concurrency: int = 8,
        timeout_s: float = 10.0,
        price_per_m_input_tokens: float = 0.0,
        max_attempts: int = 5,
        backoff_s: float = 1.0,
        backoff_max_s: float = 30.0,
        batch_questions: bool = False,
        batch_repeats: bool = False,
        mock: dict[str, float] | None = None,
    ) -> None:
        if repeats < 1 or concurrency < 1 or max_attempts < 1:
            raise ValueError("repeats, concurrency and max_attempts must be >= 1")
        self.name = name
        self.version = version
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.repeats = repeats
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.price_per_m_input_tokens = price_per_m_input_tokens
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        self.batch_questions = batch_questions
        self.batch_repeats = batch_repeats
        self.mock_generator = MockGenerator(**(mock or {}))
        self.last_errors: dict[str, str] = {}
        self._ctx: RunContext | None = None
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._jitter = wait_random_exponential(multiplier=backoff_s, max=backoff_max_s)

    # ---- adapter hooks -------------------------------------------------------------
    @property
    def endpoint(self) -> str:
        return self.base_url

    def headers(self) -> dict[str, str]:
        return {}

    def build_request(self, item: Item, questions: Mapping[str, Question]) -> dict[str, Any]:
        raise NotImplementedError

    def parse_response(
        self, raw: dict[str, Any], questions: Mapping[str, Question]
    ) -> tuple[dict[str, dict[str, float]], Usage]:
        raise NotImplementedError

    def fake_response(
        self,
        request: dict[str, Any],
        questions: Mapping[str, Question],
        probs: Mapping[str, dict[str, float]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def count_tokens(self, request: dict[str, Any]) -> int:
        raise NotImplementedError

    def price(self, usage: Usage) -> float:
        return usage.input_tokens * self.price_per_m_input_tokens / 1_000_000

    # ---- Model protocol ------------------------------------------------------------
    @property
    def model_id(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def mocked(self) -> bool:
        return self.always_mock or (self._ctx is not None and self._ctx.mode == "mock")

    def bind(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def fit(self, items: list[Item], question: Question) -> None:
        return None

    def _single_cost(self, item: Item, question: Question) -> float:
        request = self.build_request(item, {SINGLE: question})
        return self.price(Usage(self.count_tokens(request)))

    def estimate_cost(self, items: list[Item], question: Question) -> float:
        """Cost of a live, unbatched run over `items`, ignoring the cache (an upper bound)."""
        return sum(self._single_cost(i, question) for i in items) * self.repeats

    def pending_cost(self, items: list[Item], questions: Sequence[Question]) -> float:
        """Live cost of the answers not yet in this model's cache (needs bind()).

        Worst case per missing answer, as the budget reserves it. Mock runs read
        the mock cache, so their pending cost is what a mock re-run would redo.
        """
        if self._ctx is None:
            raise RuntimeError(f"{self.model_id}: call bind(RunContext) before pending_cost()")
        namespace = self._namespace()
        cache = ResponseCache(self._ctx.paths.cache_dir(self.name, self.version))
        try:
            total = 0.0
            for item in items:
                for q in questions:
                    single = self._single_cost(item, q)
                    for r in range(self.repeats):
                        if cache.get(cache_key(item.state, q, r, namespace)) is None:
                            total += single
            return total
        finally:
            cache.close()

    def estimate_cost_questions(self, items: list[Item], questions: Sequence[Question]) -> float:
        """Upper bound for predict_questions(): batching is assumed to save nothing."""
        return sum(self.estimate_cost(items, q) for q in questions)

    async def predict(self, items: list[Item], question: Question) -> list[Prediction]:
        return (await self.predict_questions(items, [question]))[0]

    async def predict_questions(
        self, items: list[Item], questions: Sequence[Question]
    ) -> list[list[Prediction]]:
        """Predictions per question, in the order given (e.g. phrasings of one task)."""
        for q in questions:
            if q.kind not in self.supports:
                raise ValueError(f"{self.model_id} does not support {q.kind!r} questions")
        if self._ctx is None:
            raise RuntimeError(f"{self.model_id}: call bind(RunContext) before predict()")
        ctx = self._ctx
        self.last_errors = {}
        cache = ResponseCache(ctx.paths.cache_dir(self.name, self.version))
        sem = asyncio.Semaphore(self.concurrency)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                tasks = [
                    asyncio.create_task(self._item(client, sem, cache, ctx, item, questions))
                    for item in items
                ]
                try:
                    per_item = await asyncio.gather(*tasks)
                except (BudgetExceeded, FatalApiError):
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
        finally:
            cache.close()
        out: list[list[Prediction]] = [[] for _ in questions]
        for preds in per_item:
            for qi, pred in enumerate(preds):
                if pred is not None:
                    out[qi].append(pred)
        return out

    # ---- internals -----------------------------------------------------------------
    def _namespace(self) -> str:
        parts: list[str] = []
        if self.mocked:
            parts.append(self.mock_generator.namespace())
        if self.batch_questions or self.batch_repeats:
            parts.append(f"batch:q={int(self.batch_questions)}:r={int(self.batch_repeats)}")
        return "|".join(parts)

    def _group(self, slots: list[Slot]) -> list[list[Slot]]:
        """Which missing answers share a request."""
        if not slots:
            return []
        if self.batch_questions and self.batch_repeats:
            return [slots]
        if self.batch_questions or self.batch_repeats:
            groups: dict[int, list[Slot]] = {}
            for s in slots:
                groups.setdefault(s.repeat if self.batch_questions else s.q_index, []).append(s)
            return list(groups.values())
        return [[s] for s in slots]

    async def _item(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        cache: ResponseCache,
        ctx: RunContext,
        item: Item,
        questions: Sequence[Question],
    ) -> list[Prediction | None]:
        namespace = self._namespace()
        results: dict[tuple[int, int], CallResult] = {}
        missing: list[Slot] = []
        for qi, q in enumerate(questions):
            for r in range(self.repeats):
                key = cache_key(item.state, q, r, namespace)
                hit = cache.get(key)
                if hit is None:
                    missing.append(Slot(qi, r, key))
                else:
                    results[(qi, r)] = CallResult(
                        probs={str(k): float(v) for k, v in hit["probs"].items()},
                        cost_usd=float(hit["cost_usd"]),
                        latency_ms=float(hit["latency_ms"]),
                    )
        groups = self._group(missing)
        outcomes = await asyncio.gather(
            *(self._request(client, sem, cache, ctx, item, questions, g) for g in groups),
            return_exceptions=True,
        )
        failed: dict[int, str] = {}
        for group, outcome in zip(groups, outcomes, strict=True):
            if isinstance(outcome, ItemFailed):
                for s in group:
                    failed[s.q_index] = str(outcome)
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                results.update(outcome)
        if failed:
            reasons = sorted(set(failed.values()))
            self.last_errors[item.id] = "; ".join(reasons)
            log.warning("%s: item %s failed: %s", self.model_id, item.id, "; ".join(reasons))
        return [
            None
            if qi in failed
            else self._aggregate(item, [results[(qi, r)] for r in range(self.repeats)])
            for qi in range(len(questions))
        ]

    def _aggregate(self, item: Item, results: list[CallResult]) -> Prediction:
        keys = list(results[0].probs)
        probs = {k: sum(r.probs[k] for r in results) / len(results) for k in keys}
        std: dict[str, float] | None = None
        if len(results) > 1:
            std = {
                k: math.sqrt(sum((r.probs[k] - probs[k]) ** 2 for r in results) / len(results))
                for k in keys
            }
        return Prediction(
            item_id=item.id,
            probs=probs,
            std=std,
            model=self.model_id,
            latency_ms=sum(r.latency_ms for r in results) / len(results),
            cost_usd=sum(r.cost_usd for r in results),
        )

    async def _request(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        cache: ResponseCache,
        ctx: RunContext,
        item: Item,
        questions: Sequence[Question],
        group: list[Slot],
    ) -> dict[tuple[int, int], CallResult]:
        """One request for every slot in `group`; caches each answer separately."""
        names = {(SINGLE if len(group) == 1 else f"q{s.q_index}r{s.repeat}"): s for s in group}
        qmap = {n: questions[s.q_index] for n, s in names.items()}
        request = self.build_request(item, qmap)
        if self.mocked:
            fake = {
                n: self.mock_generator.probs(self.name, item, qmap[n], s.repeat)
                for n, s in names.items()
            }
            raw = self.fake_response(request, qmap, fake)
            parsed = self._parse(raw, qmap)[0]
            cost, latency_ms = 0.0, 0.0
        else:
            # Reserve the worst case: as if every question were sent on its own.
            estimate = sum(self._single_cost(item, q) for q in qmap.values())
            ctx.budget.reserve(estimate)
            settled = False
            try:
                start = time.perf_counter()
                raw = await self._post(client, sem, request)
                latency_ms = (time.perf_counter() - start) * 1000
                try:
                    parsed, usage = self._parse(raw, qmap)
                except ItemFailed:
                    ctx.budget.settle(self.model_id, estimate, estimate)  # billed, unusable
                    settled = True
                    raise
                cost = self.price(usage)
                ctx.budget.settle(self.model_id, estimate, cost)
                settled = True
            finally:
                if not settled:
                    ctx.budget.settle(self.model_id, estimate, 0.0)
        share = cost / len(group)
        out: dict[tuple[int, int], CallResult] = {}
        for n, s in names.items():
            cache.set(
                s.key,
                {
                    "raw": raw,
                    "name": n,
                    "probs": parsed[n],
                    "cost_usd": share,
                    "latency_ms": latency_ms,
                },
            )
            out[(s.q_index, s.repeat)] = CallResult(parsed[n], share, latency_ms)
        return out

    def _parse(
        self, raw: dict[str, Any], questions: Mapping[str, Question]
    ) -> tuple[dict[str, dict[str, float]], Usage]:
        try:
            by_name, usage = self.parse_response(raw, questions)
            missing = set(questions) - set(by_name)
            if missing:
                raise ParseError(f"no answer for {sorted(missing)}")
            return {n: validate_probs(by_name[n], q) for n, q in questions.items()}, usage
        except ParseError as e:
            raise ItemFailed(f"unparseable response: {e}") from e

    def _wait(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome is not None else None
        if isinstance(exc, RetryableError) and exc.retry_after is not None:
            return min(exc.retry_after, self.backoff_max_s)
        return float(self._jitter(state))

    async def _post(
        self, client: httpx.AsyncClient, sem: asyncio.Semaphore, request: dict[str, Any]
    ) -> dict[str, Any]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=self._wait,
            retry=retry_if_exception_type(RetryableError),
            sleep=self._sleep,
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    async with sem:
                        try:
                            response = await client.post(
                                self.endpoint, json=request, headers=self.headers()
                            )
                        except httpx.TimeoutException as e:
                            raise RetryableError(f"timeout after {self.timeout_s}s") from e
                        except httpx.TransportError as e:
                            raise RetryableError(f"{type(e).__name__}: {e}") from e
                    return self._classify(response)
        except RetryableError as e:
            raise ItemFailed(f"gave up after {self.max_attempts} attempts: {e}") from e
        raise AssertionError("unreachable")  # pragma: no cover

    def _classify(self, response: httpx.Response) -> dict[str, Any]:
        status = response.status_code
        if status == 429 or status >= 500:
            raise RetryableError(f"HTTP {status}", _retry_after(response))
        if status >= 400:
            raise FatalApiError(f"{self.model_id}: HTTP {status}: {response.text[:500]}")
        try:
            body: Any = response.json()
        except ValueError as e:
            raise RetryableError(f"non-JSON body: {e}") from e
        if not isinstance(body, dict):
            raise RetryableError("response body is not a JSON object")
        return cast(dict[str, Any], body)
