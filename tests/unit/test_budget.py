import pytest

from recalibench.models.budget import Budget, BudgetExceeded


def test_estimate_above_cap_refused() -> None:
    b = Budget(cap_usd=1.0)
    b.check_estimate(0.99)
    with pytest.raises(BudgetExceeded, match="exceeds the budget"):
        b.check_estimate(1.01)


def test_reservations_count_against_the_cap() -> None:
    b = Budget(cap_usd=1.0)
    b.reserve(0.6)
    with pytest.raises(BudgetExceeded, match="cap"):
        b.reserve(0.5)  # 0.6 in flight + 0.5 > 1.0
    b.settle("m@1", 0.6, 0.2)
    b.reserve(0.5)
    assert b.spent_usd == pytest.approx(0.2)
    assert b.reserved_usd == pytest.approx(0.5)


def test_spend_tracked_per_model() -> None:
    b = Budget(cap_usd=10)
    for model, cost in [("a@1", 0.1), ("b@1", 0.3), ("a@1", 0.2)]:
        b.reserve(cost)
        b.settle(model, cost, cost)
    assert dict(b.spent_by_model) == pytest.approx({"a@1": 0.3, "b@1": 0.3})
    assert b.spent_usd == pytest.approx(0.6)


def test_negative_cap_rejected() -> None:
    with pytest.raises(ValueError):
        Budget(cap_usd=-1)
