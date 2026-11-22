"""What a model needs from the run, handed over by the runner via `bind()`.

Constructors stay side-effect free (`recal list` instantiates every model), so
mode, paths and the shared budget arrive here instead of from global config.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from recalibench.core.paths import Paths
from recalibench.models.budget import Budget


@dataclass(frozen=True)
class RunContext:
    paths: Paths
    budget: Budget

    @property
    def mode(self) -> str:
        return self.paths.mode


@runtime_checkable
class NeedsContext(Protocol):
    def bind(self, ctx: RunContext) -> None: ...
