"""
Collateral (variation margin) the airline would have to post on swap and
collar hedges, marked to market daily. The insurance policy needs none:
its only cash flow is the upfront premium.

Simplifications: flat forward (forward = spot each day), zero margin
threshold, no initial margin, and months drop out of the position at
month-end (intra-month averaging is ignored).
"""
import numpy as np
import pandas as pd

from .backtest import HedgeConfig, Programme, year_fractions
from .pricing import black76_call, black76_put
from .volatility import rolling_vol


def mark_to_market(prog: Programme, prices: pd.Series, vol: pd.Series, cfg: HedgeConfig) -> pd.DataFrame:
    """Daily value of the swap and collar positions from the airline's side."""
    window = prices.loc[prog.decision_date: prog.month_ends[-1]].iloc[1:]
    hedged = cfg.hedge_ratio * cfg.monthly_volume
    rows = []
    for t, F in window.items():
        remaining = prog.month_ends[prog.month_ends > t]
        if len(remaining) == 0:
            break
        T = year_fractions(t, remaining)
        sigma = float(vol.loc[:t].dropna().iloc[-1])
        disc = np.exp(-cfg.rate * T)
        swap_value = hedged * np.sum((F - prog.swap_fixed) * disc)
        collar_value = hedged * np.sum(
            black76_call(F, prog.collar_call, T, sigma, cfg.rate)
            - black76_put(F, prog.collar_put, T, sigma, cfg.rate)
        )
        rows.append((t, swap_value, collar_value))
    mtm = pd.DataFrame(rows, columns=["date", "swap_mtm", "collar_mtm"]).set_index("date")
    mtm["swap_collateral"] = (-mtm["swap_mtm"]).clip(lower=0)
    mtm["collar_collateral"] = (-mtm["collar_mtm"]).clip(lower=0)
    mtm["year"] = prog.year
    return mtm


def collateral_history(prices: pd.Series, programmes: list[Programme], cfg: HedgeConfig) -> pd.DataFrame:
    vol = rolling_vol(prices, cfg.vol_window)
    return pd.concat([mark_to_market(p, prices, vol, cfg) for p in programmes])


def peak_collateral(history: pd.DataFrame) -> pd.DataFrame:
    """Largest amount of cash posted at any point in each programme year."""
    return history.groupby("year")[["swap_collateral", "collar_collateral"]].max()
