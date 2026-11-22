"""Plug-in contracts. Pipeline code depends on these, never on concrete classes.

Rules for every implementation:
- The constructor takes `name: str` (the registry injects the config file name).
- Constructors are cheap and side-effect free: no network, no file I/O, and API
  keys are read on first use. `recal list` instantiates every plug-in.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from recalibench.core.types import Item, Prediction, Question

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@runtime_checkable
class Dataset(Protocol):
    """`load()` returns the `schema()` columns with those dtypes, including a unique,
    non-null string `id` column that stays stable across downloads."""

    name: str

    def download(self, raw_dir: Path) -> None: ...
    def load(self, raw_dir: Path) -> pd.DataFrame: ...
    def schema(self) -> dict[str, str]: ...  # expected column -> dtype


@runtime_checkable
class Model(Protocol):
    name: str
    version: str  # pinned; part of every cache key and artifact path
    supports: frozenset[str]  # e.g. {"binary", "choice"}
    trainable: bool

    def fit(self, items: list[Item], question: Question) -> None: ...  # no-op unless trainable
    async def predict(self, items: list[Item], question: Question) -> list[Prediction]: ...
    def estimate_cost(self, items: list[Item], question: Question) -> float: ...


@runtime_checkable
class Calibrator(Protocol):
    name: str
    supports: frozenset[str]  # "binary", "multiclass"

    def fit(self, P: FloatArray, y: IntArray) -> None: ...
    def predict(self, P: FloatArray) -> FloatArray: ...
