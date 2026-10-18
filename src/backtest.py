"""
Annual hedge programmes rolled through history.

Each programme is set up on the last trading day of the previous year and
covers the 12 calendar months of the next year. Everything used to set it up
(price, volatility) is known on that day, so there is no look-ahead bias.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .data import to_monthly
from .pricing import call_spread, zero_cost_put_strike
from .strategies import collar_payoff, fuel_cost, policy_payout, swap_payoff
from .volatility import rolling_vol

STRATEGIES = ["unhedged", "swap", "collar", "policy"]


@dataclass
class HedgeConfig:
    monthly_volume: float = 10_000_000  # gallons per month
    hedge_ratio: float = 0.6            # share of volume protected
    collar_call_pct: float = 0.10       # collar ceiling, % above forward
    trigger_pct: float = 0.10           # policy trigger, % above forward
    cap_pct: float = 0.50               # policy cap, % above forward
    loading: float = 0.25               # insurer's margin over fair value
    rate: float = 0.03                  # discount rate for option pricing
    vol_window: int = 252               # trailing days for volatility


@dataclass
class Programme:
    year: int
    decision_date: pd.Timestamp
    forward: float
    sigma: float
    swap_fixed: float
    collar_call: float
    collar_put: float
    trigger: float
    cap: float
    month_ends: pd.DatetimeIndex
    expiries: np.ndarray
    fair_value_per_month: np.ndarray       # per gallon, by month
    premium_per_month: np.ndarray          # per gallon, by month (with loading)


def year_fractions(start: pd.Timestamp, ends: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((ends - start).days, dtype=float) / 365.25


def set_up_programme(prices: pd.Series, vol: pd.Series, year: int, cfg: HedgeConfig) -> Programme:
    history = prices.loc[: f"{year - 1}-12-31"]
    decision_date = history.index[-1]
    forward = float(history.iloc[-1])
    sigma = float(vol.loc[:decision_date].dropna().iloc[-1])

    month_ends = pd.date_range(f"{year}-01-31", periods=12, freq="ME")
    T = year_fractions(decision_date, month_ends)

    trigger = forward * (1 + cfg.trigger_pct)
    cap = forward * (1 + cfg.cap_pct)
    collar_call = forward * (1 + cfg.collar_call_pct)
    collar_put = zero_cost_put_strike(forward, collar_call, T, sigma, cfg.rate)

    fair = call_spread(forward, trigger, cap, T, sigma, cfg.rate)
    return Programme(
        year=year, decision_date=decision_date, forward=forward, sigma=sigma,
        swap_fixed=forward, collar_call=collar_call, collar_put=collar_put,
        trigger=trigger, cap=cap, month_ends=month_ends, expiries=T,
        fair_value_per_month=fair, premium_per_month=fair * (1 + cfg.loading),
    )


def settle_programme(prog: Programme, monthly: pd.Series, cfg: HedgeConfig) -> pd.DataFrame:
    """Monthly costs of each strategy for the months that have settled."""
    settled = monthly.reindex(prog.month_ends).dropna()
    idx = prog.month_ends.get_indexer(settled.index)
    price = settled.to_numpy()
    V, h = cfg.monthly_volume, cfg.hedge_ratio

    out = pd.DataFrame(index=settled.index)
    out["year"] = prog.year
    out["avg_price"] = price
    out["budget"] = V * prog.forward  # what the month was expected to cost on decision day
    out["unhedged"] = fuel_cost(price, V, 0.0, 0.0)
    out["swap"] = fuel_cost(price, V, h, swap_payoff(price, prog.swap_fixed))
    out["collar"] = fuel_cost(price, V, h, collar_payoff(price, prog.collar_call, prog.collar_put))
    out["policy"] = fuel_cost(price, V, h, policy_payout(price, prog.trigger, prog.cap),
                              premium=prog.premium_per_month[idx])
    out["policy_payout"] = h * V * policy_payout(price, prog.trigger, prog.cap)
    out["policy_premium"] = h * V * prog.premium_per_month[idx]
    return out


def build_programmes(prices: pd.Series, cfg: HedgeConfig = HedgeConfig(),
                     start_year: int = 2008, end_year: int | None = None) -> list[Programme]:
    """Set up one hedge programme per year."""
    vol = rolling_vol(prices, cfg.vol_window)
    end_year = end_year or to_monthly(prices).index[-1].year
    return [set_up_programme(prices, vol, y, cfg) for y in range(start_year, end_year + 1)]


def programme_table(programmes: list[Programme], cfg: HedgeConfig = HedgeConfig()) -> pd.DataFrame:
    """Strikes, volatility and premiums for each year, as a table."""
    rows = []
    for prog in programmes:
        p = asdict(prog)
        for k in ("month_ends", "expiries", "fair_value_per_month", "premium_per_month"):
            p.pop(k)
        hedged = cfg.hedge_ratio * cfg.monthly_volume
        p["policy_fair_value_year"] = prog.fair_value_per_month.sum() * hedged
        p["policy_premium_year"] = prog.premium_per_month.sum() * hedged
        rows.append(p)
    return pd.DataFrame(rows).set_index("year")


def run_backtest(prices: pd.Series, cfg: HedgeConfig = HedgeConfig(),
                 start_year: int = 2008, end_year: int | None = None):
    """Returns (monthly results DataFrame, list of Programme objects)."""
    programmes = build_programmes(prices, cfg, start_year, end_year)
    monthly = to_monthly(prices)
    results = pd.concat([settle_programme(p, monthly, cfg) for p in programmes])
    results = results[results.index.notna()]
    return results, programmes


def annual_summary(results: pd.DataFrame, cfg: HedgeConfig = HedgeConfig()) -> pd.DataFrame:
    """Total cost and effective price per gallon for each strategy, by year."""
    totals = results.groupby("year")[STRATEGIES].sum()
    gallons = results.groupby("year").size() * cfg.monthly_volume
    effective = totals.div(gallons, axis=0).add_suffix("_per_gal")
    return pd.concat([totals, effective], axis=1)


def summary_metrics(results: pd.DataFrame, cfg: HedgeConfig = HedgeConfig()) -> pd.DataFrame:
    """Headline comparison across strategies over the whole backtest."""
    annual = results.groupby("year")[STRATEGIES].sum()
    full_years = results.groupby("year").size() == 12
    annual_full = annual[full_years]
    # Effectiveness is measured on the budget surprise (actual cost minus the
    # cost expected on the decision day), not on raw cost. A hedge is meant to
    # remove uncertainty about next year's bill; it cannot remove the fact that
    # fuel is dearer in some years than others.
    surprise = results[STRATEGIES].sub(results["budget"], axis=0)
    unhedged_var = surprise["unhedged"].var()

    # Downside only: months under budget count as zero, since coming in under
    # budget is not a risk the airline needs protecting from.
    overrun = surprise.clip(lower=0)
    unhedged_semivar = (overrun["unhedged"] ** 2).mean()

    metrics = {}
    for s in STRATEGIES:
        metrics[s] = {
            "total_cost_$m": results[s].sum() / 1e6,
            "avg_annual_cost_$m": annual_full[s].mean() / 1e6,
            "annual_cost_std_$m": annual_full[s].std() / 1e6,
            "worst_month_$m": results[s].max() / 1e6,
            "worst_year_$m": annual_full[s].max() / 1e6,
            "budget_surprise_std_$m": surprise[s].std() / 1e6,
            "hedge_effectiveness": 1 - surprise[s].var() / unhedged_var,
            "overrun_semidev_$m": np.sqrt((overrun[s] ** 2).mean()) / 1e6,
            "p95_overrun_$m": surprise[s].quantile(0.95) / 1e6,
            "downside_effectiveness": 1 - (overrun[s] ** 2).mean() / unhedged_semivar,
        }
    table = pd.DataFrame(metrics).T
    table.loc["policy", "premiums_paid_$m"] = results["policy_premium"].sum() / 1e6
    table.loc["policy", "payouts_received_$m"] = results["policy_payout"].sum() / 1e6
    fair_total = results["policy_premium"].sum() / (1 + cfg.loading)
    table.loc["policy", "breakeven_loading"] = results["policy_payout"].sum() / fair_total - 1
    
    return table


def years_policy_beats(results: pd.DataFrame, other: str) -> int:
    """Number of years in which the policy cost less in total than `other`."""
    annual = results.groupby("year")[["policy", other]].sum()
    return int((annual["policy"] < annual[other]).sum())
