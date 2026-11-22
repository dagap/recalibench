"""One spend cap shared by every model in a run.

Before a run: `check_estimate` refuses to start if the estimate would cross the
cap. During it: each live call `reserve`s its estimated cost first and `settle`s
the actual cost after, so concurrent calls can never jointly overshoot as long
as actual cost stays within the (deliberately conservative) estimate.
"""

from collections import defaultdict
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Spending more would cross the run's budget cap. The run stops."""


@dataclass
class Budget:
    cap_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    spent_by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def __post_init__(self) -> None:
        if self.cap_usd < 0:
            raise ValueError("budget cap must be >= 0")

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd - self.reserved_usd

    def check_estimate(self, estimate_usd: float) -> None:
        if self.spent_usd + estimate_usd > self.cap_usd:
            raise BudgetExceeded(
                f"estimated cost ${estimate_usd:.4f} (+${self.spent_usd:.4f} spent) "
                f"exceeds the budget of ${self.cap_usd:.2f}; raise budget_usd or use --limit"
            )

    def reserve(self, amount_usd: float) -> None:
        if amount_usd > self.remaining_usd:
            raise BudgetExceeded(
                f"budget cap ${self.cap_usd:.2f} reached "
                f"(spent ${self.spent_usd:.4f}, in flight ${self.reserved_usd:.4f})"
            )
        self.reserved_usd += amount_usd

    def settle(self, model: str, reserved_usd: float, actual_usd: float) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - reserved_usd)
        self.spent_usd += actual_usd
        self.spent_by_model[model] += actual_usd
