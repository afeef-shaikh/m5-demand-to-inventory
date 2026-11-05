"""
M5 Demand Forecasting → Inventory Decisions — interactive dashboard.

Pick a service level and cost assumptions; see how a forecast-driven ordering policy
compares with a naive moving-average policy on stockouts, inventory, and total cost.
Data comes from the backtest in notebooks 03–04 (Texas FOODS, 12 weeks, Feb 29 – May 22, 2016).
"""
from pathlib import Path
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

DATA = Path(__file__).parent / "data"
BLUE, GRAY, INK = "#2a78d6", "#8a8984", "#52514e"

st.set_page_config(page_title="Demand → Inventory", page_icon="📦", layout="wide")


@st.cache_data
def load():
    tradeoff = pd.read_csv(DATA / "tradeoff.csv")
    series = pd.read_parquet(DATA / "series_sample.parquet")
    summary = json.loads((DATA / "summary.json").read_text())
    return tradeoff, series, summary


tradeoff, series, summary = load()

# ---------------- Sidebar: decisions & assumptions ----------------
st.sidebar.header("Policy settings")
service = st.sidebar.slider("Target service level", 0.50, 0.995, 0.95, 0.005, format="%.3f",
                            help="Chance of not running out in a given week. Sets the safety-stock multiplier z.")
z = float(np.clip(norm.ppf(service), tradeoff["z"].min(), tradeoff["z"].max()))
st.sidebar.caption(f"Safety-stock multiplier z = {z:.2f}")

st.sidebar.header("Cost assumptions")
holding_rate = st.sidebar.slider("Annual holding cost (% of inventory value)", 5, 60, 25, 1) / 100
margin = st.sidebar.slider("Gross margin lost per unit not sold (% of price)", 5, 60, 30, 1) / 100

# ---------------- Header ----------------
st.title("📦 From demand forecast to ordering decision")
st.markdown(
    f"**Walmart M5 · Texas stores · FOODS · {summary['n_series']:,} product-store series · "
    f"12-week backtest ({summary['test_start']} → {summary['test_end']})**  \n"
    "Every product gets two deliveries a week (Mon, Thu) and is topped up to **expected demand until the next "
    "delivery + safety stock**. The *forecast policy* takes expected demand from LightGBM; the *naive policy* uses "
    "the 28-day moving average. Same rule, same safety-stock spread, same information date. Only the forecast differs."
)


def at_z(policy, zval):
    t = tradeoff[tradeoff["policy"] == policy].sort_values("z")
    return {c: float(np.interp(zval, t["z"], t[c])) for c in t.columns if c not in ("policy",)}


def daily_cost(r):
    hold = r["avg_onhand_value"] * holding_rate / 365
    lost = r["lost_value_per_day"] * margin
    return hold, lost, hold + lost


m, nv = at_z("Forecast policy", z), at_z("Naive policy", z)
mh, ml, mt = daily_cost(m)
nh, nl, nt = daily_cost(nv)

# ---------------- KPI tiles ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lost sales (forecast policy)", f"{m['lost_rate']:.1%}",
          f"{(m['lost_rate'] - nv['lost_rate']) * 100:+.1f} pts vs naive", delta_color="inverse")
c2.metric("Avg inventory value on hand", f"${m['avg_onhand_value']:,.0f}",
          f"{m['avg_onhand_value'] / nv['avg_onhand_value'] - 1:+.1%} vs naive", delta_color="inverse")
c3.metric("Total daily cost (holding + lost margin)", f"${mt:,.0f}",
          f"{mt / nt - 1:+.1%} vs naive", delta_color="inverse")
c4.metric("Fill rate (share of demand served)", f"{1 - m['lost_rate']:.1%}",
          f"{(nv['lost_rate'] - m['lost_rate']) * 100:+.1f} pts vs naive")

# ---------------- Trade-off curve ----------------
st.subheader("The trade-off: inventory held vs. sales lost")
st.caption("Each line traces one policy as the service level moves from 50% to 99.98%. "
           "Lower-left is better. Dots mark the service level chosen in the sidebar.")
curve = tradeoff.copy()
curve["Lost sales %"] = curve["lost_rate"] * 100
curve["Avg inventory value ($)"] = curve["avg_onhand_value"]
dots = pd.DataFrame([{**m, "policy": "Forecast policy"}, {**nv, "policy": "Naive policy"}])
dots["Lost sales %"] = dots["lost_rate"] * 100
dots["Avg inventory value ($)"] = dots["avg_onhand_value"]
color = alt.Color("policy:N", scale=alt.Scale(domain=["Forecast policy", "Naive policy"], range=[BLUE, GRAY]),
                  legend=alt.Legend(title=None, orient="top-right"))
tooltip = [alt.Tooltip("policy:N"), alt.Tooltip("z:Q", format=".2f"),
           alt.Tooltip("Lost sales %:Q", format=".2f"), alt.Tooltip("Avg inventory value ($):Q", format=",.0f")]
lines = alt.Chart(curve).mark_line(strokeWidth=2).encode(
    x=alt.X("Avg inventory value ($):Q", scale=alt.Scale(zero=False)),
    y=alt.Y("Lost sales %:Q"), color=color, tooltip=tooltip)
pts = alt.Chart(dots).mark_point(size=120, filled=True, stroke="white", strokeWidth=2).encode(
    x="Avg inventory value ($):Q", y="Lost sales %:Q", color=color, tooltip=tooltip)
st.altair_chart((lines + pts).properties(height=380), use_container_width=True)

hl = summary["headline"]
st.info(f"**Headline:** at the same average inventory as the naive policy at a 95% service target, the forecast "
        f"policy loses **{hl['lost_reduction_equal_inventory']:.0%} fewer sales** "
        f"({hl['naive_lost_rate']:.1%} → {hl['model_lost_rate_equal_inventory']:.1%} of demand). "
        f"To match the naive policy's lost-sales rate it needs **{hl['inventory_saving_equal_lost']:.0%} less inventory**.")

# ---------------- Cost breakdown ----------------
st.subheader("Daily cost at the chosen service level")
cost = pd.DataFrame({
    "policy": ["Forecast policy", "Forecast policy", "Naive policy", "Naive policy"],
    "component": ["Holding cost", "Lost margin", "Holding cost", "Lost margin"],
    "dollars": [mh, ml, nh, nl]})
bars = alt.Chart(cost).mark_bar().encode(
    y=alt.Y("policy:N", title=None, sort=["Forecast policy", "Naive policy"],
            scale=alt.Scale(paddingInner=0.35)),
    x=alt.X("dollars:Q", title="$ per day (all series)", stack="zero"),
    color=alt.Color("component:N", scale=alt.Scale(domain=["Holding cost", "Lost margin"], range=[BLUE, "#eb6834"]),
                    legend=alt.Legend(title=None, orient="top")),
    tooltip=["policy", "component", alt.Tooltip("dollars:Q", format="$,.0f")])
st.altair_chart(bars.properties(height=alt.Step(44)), use_container_width=True)

# ---------------- Product explorer ----------------
st.subheader("Look at one product")
labels = series[["item_id", "store_id", "label"]].drop_duplicates().sort_values("label")
options = labels["label"].tolist()
default = options.index("FOODS_3_586 · TX_2") if "FOODS_3_586 · TX_2" in options else 0
pick = st.selectbox("Product-store", options, index=default)
one = series[series["label"] == pick].sort_values("date")
base = alt.Chart(one).encode(x=alt.X("date:T", title=None))
band = base.mark_area(opacity=0.18, color=BLUE).encode(
    y=alt.Y("p50:Q", title="Units per day"), y2="p90:Q")
actual = base.mark_line(color=INK, strokeWidth=1.5).encode(y="sales:Q", tooltip=["date:T", "sales:Q"])
fc = base.mark_line(color=BLUE, strokeWidth=2).encode(y="mean:Q", tooltip=[alt.Tooltip("mean:Q", format=".1f")])
st.altair_chart((band + actual + fc).properties(height=280), use_container_width=True)
st.caption("Dark line: actual sales · Blue line: forecast (expected demand) · Shaded band: P50–P90 forecast range. "
           "Forecasts for each 4-week block were made on the Monday the block starts. "
           "The band is shown for risk context; safety stock uses the 28-day spread (see Notebook 04).")

with st.expander("Method & assumptions"):
    st.markdown(summary["method_md"])
