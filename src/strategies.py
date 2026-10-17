"""
Settlement payoffs for each strategy, per gallon of hedged volume.

Sign convention: a positive payoff is money received by the airline, which
reduces its fuel bill. `price` is the month's average jet fuel price.
"""
import numpy as np


def swap_payoff(price, fixed):
    """Swap: the airline receives (market - fixed). Negative when prices fall."""
    return np.asarray(price) - fixed


def collar_payoff(price, call_strike, put_strike):
    """
    Collar: long call (protection above call_strike), short put
    (the airline pays the shortfall below put_strike).
    """
    price = np.asarray(price)
    return np.maximum(price - call_strike, 0.0) - np.maximum(put_strike - price, 0.0)


def policy_payout(price, trigger, cap):
    """Trigger-cap insurance: pays the rise above trigger, up to cap. Never negative."""
    price = np.asarray(price)
    return np.clip(price - trigger, 0.0, cap - trigger)


def fuel_cost(price, volume, hedge_ratio, payoff, premium=0.0):
    """
    Total cost for one month: buy all fuel at market, then net off the hedge
    payoff on the hedged share of volume, plus any premium for that month.
    """
    hedged = hedge_ratio * volume
    return np.asarray(price) * volume - hedged * np.asarray(payoff) + hedged * premium
