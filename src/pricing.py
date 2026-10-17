"""Black-76 option pricing and the products built from it."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(F, K, T, sigma):
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return d1, d1 - sd


def black76_call(F, K, T, sigma, r=0.0):
    """Price of a call on a forward/futures price F (Black-76)."""
    F, K, T = np.asarray(F, float), np.asarray(K, float), np.asarray(T, float)
    intrinsic = np.maximum(F - K, 0.0)
    Ts = np.maximum(T, 1e-12)
    d1, d2 = _d1_d2(F, K, Ts, sigma)
    price = np.exp(-r * Ts) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.where(T <= 0, intrinsic, price)


def black76_put(F, K, T, sigma, r=0.0):
    """Price of a put on a forward/futures price F (Black-76)."""
    F, K, T = np.asarray(F, float), np.asarray(K, float), np.asarray(T, float)
    intrinsic = np.maximum(K - F, 0.0)
    Ts = np.maximum(T, 1e-12)
    d1, d2 = _d1_d2(F, K, Ts, sigma)
    price = np.exp(-r * Ts) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(T <= 0, intrinsic, price)


def call_spread(F, K_low, K_high, T, sigma, r=0.0):
    """Long call at K_low, short call at K_high: the trigger-cap policy payoff."""
    return black76_call(F, K_low, T, sigma, r) - black76_call(F, K_high, T, sigma, r)


def policy_premium(F, trigger, cap, expiries, sigma, r=0.0, loading=0.25):
    """
    Upfront premium per gallon of monthly volume for a strip of monthly
    trigger-cap payouts. Fair value = sum of call spreads; the premium charged
    adds the insurer's loading. Returns (fair_value, premium_charged).
    """
    fair = float(np.sum(call_spread(F, trigger, cap, expiries, sigma, r)))
    return fair, fair * (1.0 + loading)


def zero_cost_put_strike(F, call_strike, expiries, sigma, r=0.0):
    """
    Put strike that makes a strip of collars cost nothing upfront:
    the premiums of the calls bought equal the premiums of the puts sold.
    """
    call_value = float(np.sum(black76_call(F, call_strike, expiries, sigma, r)))

    def gap(k_put):
        return float(np.sum(black76_put(F, k_put, expiries, sigma, r))) - call_value

    return brentq(gap, 1e-6 * F, F)
