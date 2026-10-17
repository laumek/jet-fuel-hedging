import numpy as np
import pytest

from src.strategies import collar_payoff, fuel_cost, policy_payout, swap_payoff


def test_swap_fixes_the_price():
    prices = np.array([2.0, 3.0, 4.0])
    cost = fuel_cost(prices, 100, 1.0, swap_payoff(prices, 3.0))
    assert np.allclose(cost, 300)


def test_policy_pays_between_trigger_and_cap():
    payout = policy_payout(np.array([3.0, 3.5, 4.0, 6.0]), trigger=3.3, cap=4.5)
    assert np.allclose(payout, [0.0, 0.2, 0.7, 1.2])


def test_collar_keeps_price_in_band():
    prices = np.array([1.5, 2.5, 4.0])
    cost_per_gal = fuel_cost(prices, 1, 1.0, collar_payoff(prices, 3.3, 2.0))
    assert np.allclose(cost_per_gal, [2.0, 2.5, 3.3])


def test_premium_added_to_cost():
    cost = fuel_cost(3.0, 100, 0.5, 0.0, premium=0.1)
    assert cost == pytest.approx(300 + 50 * 0.1)
