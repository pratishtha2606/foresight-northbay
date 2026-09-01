"""
Project FORESIGHT — D5 Planning dashboard.

Run locally:
    streamlit run app/dashboard.py

Deploy free on Streamlit Community Cloud:
    1. Push this repo to GitHub (public or private).
    2. Go to https://share.streamlit.io -> "New app".
    3. Point it at app/dashboard.py, set requirements.txt as the deps file.
    4. Deploy — you'll get a public URL for D6/D5 submission.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="FORESIGHT — NorthBay Living", layout="wide")


@st.cache_data
def load_data():
    risk = pd.read_csv(PROC / "risk_scored.csv")
    forecasts = pd.read_csv(PROC / "forecasts.csv", parse_dates=["week_start"])
    weekly = pd.read_csv(PROC / "weekly_sku_demand.csv", parse_dates=["week_start"])
    backtest = pd.read_csv(PROC / "backtest_results.csv")
    return risk, forecasts, weekly, backtest


st.title("📦 FORESIGHT — Demand & Inventory Intelligence")
st.caption("NorthBay Living · Planning Dashboard")

try:
    risk, forecasts, weekly, backtest = load_data()
except FileNotFoundError:
    st.error("Processed data not found. Run the pipeline first: "
             "`python src/pipeline.py && python src/forecast.py && python src/risk.py`")
    st.stop()

if risk.empty:
    st.warning("No SKUs scored yet — check the pipeline output.")
    st.stop()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
categories = ["All"] + sorted(risk["category"].dropna().unique().tolist())
sel_cat = st.sidebar.selectbox("Category", categories)
quadrants = ["All"] + sorted(risk["risk_quadrant"].unique().tolist())
sel_quad = st.sidebar.selectbox("Risk quadrant", quadrants)
sku_search = st.sidebar.text_input("Search SKU ID")

f = risk.copy()
if sel_cat != "All":
    f = f[f["category"] == sel_cat]
if sel_quad != "All":
    f = f[f["risk_quadrant"] == sel_quad]
if sku_search:
    f = f[f["sku_id"].str.contains(sku_search.strip(), case=False)]

# ---------------- Headline KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs monitored", f"{len(risk):,}")
c2.metric("Reorder Now", int((risk["risk_quadrant"] == "Reorder Now").sum()))
c3.metric("Sales at risk (₹)", f"{risk['sales_at_risk_rupees'].sum():,.0f}")
c4.metric("Capital locked (₹)", f"{risk['capital_locked_rupees'].sum():,.0f}")

model_wape = backtest["wape_model"].mean()
base_wape = backtest["wape_baseline"].mean()
st.caption(f"Backtested forecast accuracy — model WAPE **{model_wape:.1%}** vs "
           f"seasonal-naive baseline **{base_wape:.1%}** "
           f"({(base_wape - model_wape) / base_wape:.0%} improvement).")

st.divider()

# ---------------- Decisioning grid ----------------
st.subheader("Decisioning view — stockout vs overstock risk")
fig = px.scatter(f, x="overstock_risk", y="stockout_risk", size="revenue_at_stake",
                  color="risk_quadrant", hover_data=["sku_id", "category", "recommended_action"],
                  color_discrete_map={"Reorder Now": "#d64545", "Markdown / Clear": "#5b6fd6",
                                       "Watch / Volatile": "#d6a545", "Healthy": "#4caf7d"},
                  labels={"overstock_risk": "Overstock risk →", "stockout_risk": "Stockout risk →"})
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- Prioritised action list ----------------
st.subheader("Prioritised reorder / markdown list")
if f.empty:
    st.info("No SKUs match the current filters.")
else:
    show_cols = ["sku_id", "category", "risk_quadrant", "recommended_action",
                 "weeks_of_cover", "stockout_risk", "overstock_risk", "revenue_at_stake"]
    st.dataframe(
        f[show_cols].sort_values("revenue_at_stake", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "stockout_risk": st.column_config.ProgressColumn("Stockout risk", min_value=0, max_value=1),
            "overstock_risk": st.column_config.ProgressColumn("Overstock risk", min_value=0, max_value=1),
            "revenue_at_stake": st.column_config.NumberColumn("₹ at stake", format="₹%.0f"),
        },
    )

st.divider()

# ---------------- SKU drill-down ----------------
st.subheader("Forecast vs actual — SKU drill-down")
sku_options = sorted(f["sku_id"].unique()) if not f.empty else sorted(risk["sku_id"].unique())
if sku_options:
    sel_sku = st.selectbox("Choose a SKU", sku_options)
    hist = weekly[weekly["sku_id"] == sel_sku].sort_values("week_start").tail(26)
    fc = forecasts[forecasts["sku_id"] == sel_sku].sort_values("week_start")

    if hist.empty:
        st.info("No history for this SKU.")
    else:
        fig2 = px.line(hist, x="week_start", y="units_sold", labels={"units_sold": "Units/week"})
        fig2.update_traces(name="Actual", showlegend=True)
        if not fc.empty:
            fig2.add_scatter(x=fc["week_start"], y=fc["forecast_units"], mode="lines+markers",
                              name="Forecast", line=dict(color="#5b6fd6"))
            fig2.add_scatter(x=pd.concat([fc["week_start"], fc["week_start"][::-1]]),
                              y=pd.concat([fc["forecast_upper"], fc["forecast_lower"][::-1]]),
                              fill="toself", fillcolor="rgba(91,111,214,0.15)",
                              line=dict(color="rgba(255,255,255,0)"), name="Uncertainty band",
                              showlegend=True)
        fig2.update_layout(height=400, legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

        row = risk[risk["sku_id"] == sel_sku]
        if not row.empty:
            r = row.iloc[0]
            st.info(f"**{sel_sku}** ({r['category']}) — {r['risk_quadrant']}: {r['recommended_action']} "
                    f"· Weeks of cover: {r['weeks_of_cover']:.1f} · ₹{r['revenue_at_stake']:,.0f} at stake")
else:
    st.info("No SKUs available for drill-down with the current filters.")
