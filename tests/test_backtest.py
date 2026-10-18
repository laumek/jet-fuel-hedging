import numpy as np
import pandas as pd
import pytest

from src.backtest import HedgeConfig, run_backtest, summary_metrics
from src.margin import collateral_history, peak_collateral


def synthetic_prices(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2007-01-01", "2012-12-31")
    returns = rng.normal(0, 0.4 / np.sqrt(252), len(dates))
    return pd.Series(2.5 * np.exp(np.cumsum(returns)), index=dates, name="price")


def test_backtest_runs_and_unhedged_is_baseline():
    prices = synthetic_prices()
    results, programmes = run_backtest(prices, HedgeConfig(), 2008, 2012)
    assert len(programmes) == 5
    assert len(results) == 60
    metrics = summary_metrics(results)
    assert metrics.loc["unhedged", "hedge_effectiveness"] == 0
    assert metrics.loc["swap", "hedge_effectiveness"] > 0


def test_no_look_ahead_in_setup():
    prices = synthetic_prices()
    _, programmes = run_backtest(prices, HedgeConfig(), 2009, 2009)
    assert programmes[0].decision_date < pd.Timestamp("2009-01-01")


def test_collateral_is_never_negative():
    prices = synthetic_prices()
    cfg = HedgeConfig()
    _, programmes = run_backtest(prices, cfg, 2008, 2009)
    peaks = peak_collateral(collateral_history(prices, programmes, cfg))
    assert (peaks >= 0).all().all()


def test_swap_effectiveness_matches_theory():
    cfg = HedgeConfig(hedge_ratio=0.6)
    results, _ = run_backtest(synthetic_prices(), cfg, 2008, 2012)
    m = summary_metrics(results, cfg)
    expected = 1 - (1 - cfg.hedge_ratio) ** 2
    assert m.loc["swap", "hedge_effectiveness"] == pytest.approx(expected)
    assert m.loc["swap", "downside_effectiveness"] == pytest.approx(expected)