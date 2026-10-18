"""
Run the full analysis and save tables and charts to results/.

Usage (from the repo root):
    python -m scripts.run_backtest
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import (HedgeConfig, STRATEGIES, annual_summary, programme_table,
                          run_backtest, summary_metrics, years_policy_beats)
from src.data import count_missing, find_gaps, load_jet_fuel, to_monthly
from src.margin import collateral_history, peak_collateral
from src.volatility import annualised_vol, rolling_vol

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVENTS = {
    "2008 spike & crash": ("2008-01-01", "2008-12-31"),
    "2014–16 collapse": ("2014-07-01", "2016-02-29"),
    "COVID": ("2020-02-15", "2020-06-30"),
    "Ukraine war": ("2022-02-24", "2022-12-31"),
    "2026 spike": ("2026-01-01", "2026-10-31"),
}
COLOURS = {"unhedged": "#7a7a7a", "swap": "#1f497d", "collar": "#c0504d", "policy": "#e8a33d"}


def shade_events(ax):
    for start, end in EVENTS.values():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#1f497d", alpha=0.07)


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg = HedgeConfig()

    # --- Data -------------------------------------------------------------
    prices = load_jet_fuel()
    monthly = to_monthly(prices)
    vol = rolling_vol(prices, cfg.vol_window)
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    print(f"Missing before cleaning: {count_missing()}; gaps of 5+ days: {len(find_gaps(prices))}")
    print(f"Min {prices.min():.3f} on {prices.idxmin().date()}, max {prices.max():.3f} on {prices.idxmax().date()}")
    print(f"Annualised volatility, whole period: {annualised_vol(prices):.1%}\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(prices, lw=0.6, color="#9aa7b8", label="Daily")
    ax1.plot(monthly, lw=1.6, color="#1f497d", label="Monthly average")
    shade_events(ax1)
    ax1.set_ylabel("$/gallon")
    ax1.set_title("US Gulf Coast jet fuel price")
    ax1.legend(frameon=False)
    ax2.plot(vol, color="#c0504d", lw=1)
    shade_events(ax2)
    ax2.set_ylabel("Annualised volatility")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(RESULTS / "01_price_and_volatility.png", dpi=150)

    # --- Backtest ---------------------------------------------------------
    results, programmes = run_backtest(prices, cfg)
    params = programme_table(programmes, cfg)
    annual = annual_summary(results, cfg)
    metrics = summary_metrics(results, cfg)

    results.to_csv(RESULTS / "monthly_results.csv")
    params.to_csv(RESULTS / "programmes.csv")
    annual.to_csv(RESULTS / "annual_summary.csv")
    metrics.to_csv(RESULTS / "summary_metrics.csv")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("Summary metrics:\n", metrics.round(2), "\n")
    n_years = results["year"].nunique()
    for other in ("unhedged", "swap", "collar"):
        print(f"Policy cheaper than {other} in {years_policy_beats(results, other)} of {n_years} years")

    fig, ax = plt.subplots(figsize=(11, 5))
    for s in STRATEGIES:
        ax.plot(annual.index, annual[f"{s}_per_gal"], marker="o", ms=3, label=s, color=COLOURS[s])
    ax.set_ylabel("Effective fuel price, $/gallon")
    ax.set_title("Effective annual fuel price by strategy")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "02_effective_price_by_strategy.png", dpi=150)

    rows = []
    for loading in (0.0, 0.25, 0.5):
        for trigger in (0.0, 0.1, 0.2):
            c = HedgeConfig(loading=loading, trigger_pct=trigger)
            res, _ = run_backtest(prices, c)
            m = summary_metrics(res, c)
            rows.append({
                "loading": loading, "trigger_pct": trigger,
                "policy_beats_swap_years": years_policy_beats(res, "swap"),
                "downside_effectiveness": m.loc["policy", "downside_effectiveness"],
                "net_payout_$m": (res["policy_payout"].sum() - res["policy_premium"].sum()) / 1e6,
            })
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(RESULTS / "sensitivity.csv", index=False)
    print("\nSensitivity:\n", sensitivity.round(2))


    # --- Margin and liquidity ---------------------------------------------
    history = collateral_history(prices, programmes, cfg)
    peaks = peak_collateral(history)
    peaks["policy_premium"] = params["policy_premium_year"]
    peaks.to_csv(RESULTS / "peak_collateral.csv")
    print("\nPeak collateral posted ($m):\n", (peaks / 1e6).round(1))

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.27
    x = peaks.index.to_numpy()
    ax.bar(x - width, peaks["swap_collateral"] / 1e6, width, label="Swap: peak collateral", color=COLOURS["swap"])
    ax.bar(x, peaks["collar_collateral"] / 1e6, width, label="Collar: peak collateral", color=COLOURS["collar"])
    ax.bar(x + width, peaks["policy_premium"] / 1e6, width, label="Policy: upfront premium", color=COLOURS["policy"])
    ax.set_ylabel("$m")
    ax.set_title("Cash tied up: peak collateral vs insurance premium")
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "03_collateral_vs_premium.png", dpi=150)
    print(f"\nSaved tables and charts to {RESULTS}")


if __name__ == "__main__":
    main()
