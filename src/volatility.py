"""Realised volatility measures."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252 #standard market convention

def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna().rename("log_return")


def annualised_vol(prices: pd.Series) -> float:
    """Annualised volatility over the whole series."""
    return float(log_returns(prices).std() * np.sqrt(TRADING_DAYS))


def rolling_vol(prices: pd.Series, window: int = TRADING_DAYS, min_periods: int | None = None) -> pd.Series:
    """
    Rolling annualised volatility over the trailing `window` observations.
    At each date it uses only past data, so it can be used for pricing
    without look-ahead bias. By default it needs 80% of the window to produce
    a value, since holidays mean a calendar year has slightly under 252 prices.
    """
    min_periods = min_periods or int(0.8 * window)
    vol = log_returns(prices).rolling(window, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS)
    return vol.rename("rolling_vol")
