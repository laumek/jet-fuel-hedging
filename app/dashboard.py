"""
Interactive comparison of fuel hedging strategies.

Run from the repo root:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import (HedgeConfig, STRATEGIES, annual_summary, programme_table,
                          run_backtest, summary_metrics, years_policy_beats)
from src.data import load_jet_fuel
from src.margin import collateral_history, peak_collateral

LABELS = {"unhedged": "Unhedged", "swap": "Swap", "collar": "Zero-cost collar", "policy": "Insurance policy"}
COLOURS = {"unhedged": "#8c8c8c", "swap": "#1f497d", "collar": "#c0504d", "policy": "#e8a33d"}

st.set_page_config(page_title="Fuel hedging comparison", layout="wide")


@st.cache_data
def get_prices():
    return load_jet_fuel()


@st.cache_data
def run(cfg_dict):
    cfg = HedgeConfig(**cfg_dict)
    prices = get_prices()
    results, programmes = run_backtest(prices, cfg)
    peaks = peak_collateral(collateral_history(prices, programmes, cfg))
    return results, programme_table(programmes, cfg), peaks


st.title("How would an airline have done with each fuel protection strategy?")
st.caption("US Gulf Coast jet fuel, 2008 to today. Each year's protection is set up on the "
           "last trading day of the year before, using only what was known then.")

with st.sidebar:
    st.header("Airline and policy settings")
    volume = st.number_input("Fuel bought per month (million gallons)", 1.0, 200.0, 10.0, 1.0)
    hedge_ratio = st.slider("Share of fuel protected", 0.0, 1.0, 0.6, 0.05)
    st.subheader("Insurance policy")
    trigger = st.slider("Pays out above (% over starting price)", 0, 50, 10, 5)
    cap = st.slider("Stops paying above (% over starting price)", trigger + 5, 150, max(50, trigger + 5), 5)
    loading = st.slider("Insurer's margin over fair price (%)", 0, 100, 25, 5)
    st.subheader("Zero-cost collar")
    collar_call = st.slider("Ceiling (% over starting price)", 0, 50, 10, 5)

cfg = dict(monthly_volume=volume * 1e6, hedge_ratio=hedge_ratio, trigger_pct=trigger / 100,
           cap_pct=cap / 100, loading=loading / 100, collar_call_pct=collar_call / 100)

try:
    results, params, peaks = run(cfg)
except FileNotFoundError:
    st.error("Price data not found. Download DJFUELUSGULF from FRED as a CSV and save it in data/.")
    st.stop()

annual = annual_summary(results, HedgeConfig(**cfg))
metrics = summary_metrics(results, HedgeConfig(**cfg))
n_years = results["year"].nunique()

cols = st.columns(4)
cols[0].metric("Policy cheaper than a swap", f"{years_policy_beats(results, 'swap')} of {n_years} years")
cols[1].metric("Policy cheaper than a collar", f"{years_policy_beats(results, 'collar')} of {n_years} years")
cols[2].metric("Largest swap collateral call", f"${peaks['swap_collateral'].max() / 1e6:,.0f}m",
               help=f"In {int(peaks['swap_collateral'].idxmax())}")
cols[3].metric("Largest policy premium", f"${params['policy_premium_year'].max() / 1e6:,.0f}m",
               help="Paid once, upfront. No further cash calls.")
net_payout = (results["policy_payout"].sum() - results["policy_premium"].sum()) / 1e6
cols2 = st.columns(3)
cols2[0].metric("Policy payouts minus premiums", f"${net_payout:+,.0f}m",
                help="Positive: the airline came out ahead on the policy. Negative: the insurer did.")
cols2[1].metric("Insurer's break-even loading", f"{metrics.loc['policy', 'breakeven_loading']:.0%}",
                help="The margin over fair value the insurer needed for premiums to cover payouts.")
cols2[2].metric("Overrun risk removed by the policy",
                f"{metrics.loc['policy', 'downside_effectiveness']:.0%}")

st.subheader("Effective fuel price each year")
fig = go.Figure()
for s in STRATEGIES:
    fig.add_trace(go.Scatter(x=annual.index, y=annual[f"{s}_per_gal"], name=LABELS[s],
                             mode="lines+markers", line=dict(color=COLOURS[s])))
fig.update_layout(yaxis_title="$ per gallon", hovermode="x unified", height=420, margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

st.subheader("Cash the airline has to find")
fig = go.Figure()
fig.add_trace(go.Bar(x=peaks.index, y=peaks["swap_collateral"] / 1e6, name="Swap: peak collateral",
                     marker_color=COLOURS["swap"]))
fig.add_trace(go.Bar(x=peaks.index, y=peaks["collar_collateral"] / 1e6, name="Collar: peak collateral",
                     marker_color=COLOURS["collar"]))
fig.add_trace(go.Bar(x=params.index, y=params["policy_premium_year"] / 1e6, name="Policy: upfront premium",
                     marker_color=COLOURS["policy"]))
fig.update_layout(barmode="group", yaxis_title="$ million", height=420, margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

st.subheader("Summary over the whole period")
table = metrics.rename(index=LABELS)[["avg_annual_cost_$m", "worst_year_$m", "p95_overrun_$m",
                                      "hedge_effectiveness", "downside_effectiveness"]]
table.columns = ["Average yearly cost ($m)", "Worst year ($m)", "Bad-month overrun ($m)",
                 "Risk removed", "Overrun risk removed"]
st.dataframe(table.style.format({"Average yearly cost ($m)": "{:,.0f}", "Worst year ($m)": "{:,.0f}",
                                 "Bad-month overrun ($m)": "{:,.1f}", "Risk removed": "{:.0%}",
                                 "Overrun risk removed": "{:.0%}"}),
             width="stretch")
st.caption("Bad-month overrun = the 95th-percentile monthly overrun against budget. "
           "Overrun risk removed counts only months over budget.")

with st.expander("Assumptions behind these numbers"):
    st.markdown(
        "- The forward price for every month is the spot price on the day protection is set up "
        "(free historical jet fuel forward curves aren't available).\n"
        "- Policy and collar are priced with Black-76 using the past year's realised volatility.\n"
        "- Collateral assumes daily margining with no threshold and no initial margin.\n"
        "- Each year's protection is set up once, on the last trading day of the year before. "
        "Results can differ with a different decision date.\n"
        "- 2026 is a partial year (January to September).\n"
        "- Figures are illustrative and do not include bank fees or credit charges."
    )
