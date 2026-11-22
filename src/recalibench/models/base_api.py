"""ApiModel: the base class for hosted models.

An adapter (e.g. jev.py) implements only the wire format:
    build_request(item, question) -> dict          the JSON body for one call
    parse_response(raw, question) -> (probs, Usage) probs keyed by question.keys
    fake_response(request, question, probs) -> dict a reply shaped like the real API
    count_tokens(request) -> int                    estimate, for the pre-run budget check
    price(usage) -> float                           USD for one call
    headers() -> dict                               auth etc. (read keys lazily here)

The base supplies: bounded concurrency, `repeats` calls per item averaged into
probs (with their std), retries with jitter on 429/5xx/timeouts, the response
cache, the shared budget, cost and latency tracking, per-item error isolation,
and mock routing: with mode=mock (or `always_mock`) the network call is replaced
by MockGenerator + fake_response, so build/parse code still runs offline.

Items that fail (retries exhausted, unparseable reply) are logged, recorded in
`last_errors` and left out of predict()'s result. Fatal API errors (401, 400,
422, ...) and BudgetExceeded stop the whole call.
"""

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
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


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return None if value is None else max(0.0, float(value))
    except ValueError:
        return None


def validate_probs(probs: dict[str, float], question: Question) -> dict[str, float]:
    """Exactly the question's keys, finite, in [0, 1], summing to 1 (renormalised)."""
    keys = set(question.keys)
    if set(probs) != keys:
        raise ParseError(f"expected probabilities for {sorted(keys)}, got {sorted(probs)}")
    values = list(probs.values())
    if any(not math.isfinite(v) or v < -1e-9 or v > 1 + 1e-9 for v in values):
        raise ParseError(f"probabilities out of range: {probs}")
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

    def build_request(self, item: Item, question: Question) -> dict[str, Any]:
        raise NotImplementedError

    def parse_response(
        self, raw: dict[str, Any], question: Question
    ) -> tuple[dict[str, float], Usage]:
        raise NotImplementedError

    def fake_response(
        self, request: dict[str, Any], question: Question, probs: dict[str, float]
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

    def estimate_cost(self, items: list[Item], question: Question) -> float:
        """Cost of a live run over `items`, ignoring the cache (an upper bound)."""
        per_item = (
            self.price(Usage(self.count_tokens(self.build_request(i, question)))) for i in items
        )
        return sum(per_item) * self.repeats

    async def predict(self, items: list[Item], question: Question) -> list[Prediction]:
        if question.kind not in self.supports:
            raise ValueError(f"{self.model_id} does not support {question.kind!r} questions")
        if self._ctx is None:
            raise RuntimeError(f"{self.model_id}: call bind(RunContext) before predict()")
        ctx = self._ctx
        self.last_errors = {}
        cache = ResponseCache(ctx.paths.cache_dir(self.name, self.version))
        sem = asyncio.Semaphore(self.concurrency)
        namespace = self.mock_generator.namespace() if self.mocked else ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:

                async def one(item: Item, r: int) -> CallResult:
                    return await self._call(client, sem, cache, ctx, item, question, r, namespace)

                async def per_item(item: Item) -> Prediction | None:
                    try:
                        results = await asyncio.gather(*(one(item, r) for r in range(self.repeats)))
                    except ItemFailed as e:
                        self.last_errors[item.id] = str(e)
                        log.warning("%s: item %s failed: %s", self.model_id, item.id, e)
                        return None
                    return self._aggregate(item, list(results))

                tasks = [asyncio.create_task(per_item(i)) for i in items]
                try:
                    done = await asyncio.gather(*tasks)
                except (BudgetExceeded, FatalApiError):
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
        finally:
            cache.close()
        return [p for p in done if p is not None]

    # ---- internals -----------------------------------------------------------------
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

    async def _call(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        cache: ResponseCache,
        ctx: RunContext,
        item: Item,
        question: Question,
        repeat_idx: int,
        namespace: str,
    ) -> CallResult:
        key = cache_key(item.state, question, repeat_idx, namespace)
        hit = cache.get(key)
        if hit is not None:
            return CallResult(
                probs={str(k): float(v) for k, v in hit["probs"].items()},
                cost_usd=float(hit["cost_usd"]),
                latency_ms=float(hit["latency_ms"]),
            )
        request = self.build_request(item, question)
        if self.mocked:
            probs = self.mock_generator.probs(self.name, item, question, repeat_idx)
            raw = self.fake_response(request, question, probs)
            latency_ms, cost = 0.0, 0.0
            parsed = self._parse(raw, question, item)[0]
        else:
            estimate = self.price(Usage(self.count_tokens(request)))
            ctx.budget.reserve(estimate)
            settled = False
            try:
                start = time.perf_counter()
                raw = await self._post(client, sem, request, item)
                latency_ms = (time.perf_counter() - start) * 1000
                try:
                    parsed, usage = self._parse(raw, question, item)
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
        cache.set(key, {"raw": raw, "probs": parsed, "cost_usd": cost, "latency_ms": latency_ms})
        return CallResult(parsed, cost, latency_ms)

    def _parse(
        self, raw: dict[str, Any], question: Question, item: Item
    ) -> tuple[dict[str, float], Usage]:
        try:
            probs, usage = self.parse_response(raw, question)
            return validate_probs(probs, question), usage
        except ParseError as e:
            raise ItemFailed(f"unparseable response: {e}") from e

    def _wait(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome is not None else None
        if isinstance(exc, RetryableError) and exc.retry_after is not None:
            return min(exc.retry_after, self.backoff_max_s)
        return float(self._jitter(state))

    async def _post(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        request: dict[str, Any],
        item: Item,
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
