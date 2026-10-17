import numpy as np
import pytest

from src.pricing import black76_call, black76_put, call_spread, policy_premium, zero_cost_put_strike

F, K, T, SIGMA, R = 3.0, 3.3, 0.5, 0.4, 0.03


def test_put_call_parity():
    lhs = black76_call(F, K, T, SIGMA, R) - black76_put(F, K, T, SIGMA, R)
    assert lhs == pytest.approx(np.exp(-R * T) * (F - K))


def test_expired_option_is_intrinsic():
    assert black76_call(3.5, 3.0, 0.0, SIGMA) == pytest.approx(0.5)
    assert black76_put(3.5, 3.0, 0.0, SIGMA) == pytest.approx(0.0)


def test_call_spread_bounds():
    value = call_spread(F, 3.3, 4.5, T, SIGMA, R)
    assert 0 < value < np.exp(-R * T) * (4.5 - 3.3)


def test_higher_vol_makes_insurance_dearer():
    assert call_spread(F, 3.3, 4.5, T, 0.6) > call_spread(F, 3.3, 4.5, T, 0.2)


def test_zero_cost_collar_costs_nothing():
    expiries = np.arange(1, 13) / 12
    k_call = F * 1.1
    k_put = zero_cost_put_strike(F, k_call, expiries, SIGMA, R)
    calls = black76_call(F, k_call, expiries, SIGMA, R).sum()
    puts = black76_put(F, k_put, expiries, SIGMA, R).sum()
    assert k_put < F
    assert calls == pytest.approx(puts, rel=1e-6)


def test_loading_applied():
    fair, charged = policy_premium(F, 3.3, 4.5, [0.5], SIGMA, loading=0.25)
    assert charged == pytest.approx(fair * 1.25)
