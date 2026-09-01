"""
Project FORESIGHT — D4 Risk scoring.

Combines the forecast with the current inventory position to flag stockout
and overstock risk per SKU, with a recommended action and rupee value at
stake (Section 08 of the brief). Transparent, rule-based — no black box.

Run:
    python src/risk.py
Output:
    data/processed/risk_scored.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

SAFETY_SERVICE_LEVEL_Z = 0.84   # ~80% service level (z-score) for the safety buffer
OVERSTOCK_COVER_WEEKS = 8       # holding more than this many weeks of forward demand = overstock signal


def score():
    forecasts = pd.read_csv(PROC / "forecasts.csv", parse_dates=["week_start"])
    inv = pd.read_csv(PROC / "inventory_latest.csv", parse_dates=["date"])
    sku_master = pd.read_csv(PROC / "sku_master_clean.csv")

    # weekly forecast demand -> convert to a daily rate for lead-time comparison
    agg = forecasts.groupby("sku_id").agg(
        forecast_avg_weekly=("forecast_units", "mean"),
        forecast_horizon_units=("forecast_units", "sum"),
        forecast_upper_weekly=("forecast_upper", "mean"),
    ).reset_index()
    agg["forecast_daily_rate"] = agg["forecast_avg_weekly"] / 7

    df = agg.merge(inv[["sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]],
                    on="sku_id", how="left")
    df = df.merge(sku_master[["sku_id", "category", "subcategory", "unit_cost", "list_price"]],
                   on="sku_id", how="left")
    df = df.dropna(subset=["on_hand_units", "lead_time_days"])

    # --- Stockout risk: projected stock over the lead time vs a safety buffer ---
    demand_over_lead_time = df["forecast_daily_rate"] * df["lead_time_days"]
    demand_std_over_lead_time = demand_over_lead_time * 0.35  # from backtest error spread (~15-16% WAPE, widened for daily noise)
    safety_stock = SAFETY_SERVICE_LEVEL_Z * demand_std_over_lead_time
    projected_position = df["on_hand_units"] + df["on_order_units"] - demand_over_lead_time
    df["stockout_gap_units"] = safety_stock - projected_position
    df["stockout_risk"] = (df["stockout_gap_units"] / (demand_over_lead_time + 1e-6)).clip(0, 1)
    df.loc[demand_over_lead_time < 1e-6, "stockout_risk"] = 0.0

    # --- Overstock risk: on-hand cover in weeks of forward demand ---
    weeks_of_cover = df["on_hand_units"] / df["forecast_avg_weekly"].replace(0, np.nan)
    df["weeks_of_cover"] = weeks_of_cover.fillna(999)
    df["overstock_risk"] = ((df["weeks_of_cover"] - OVERSTOCK_COVER_WEEKS) / OVERSTOCK_COVER_WEEKS).clip(0, 1)
    df.loc[df["forecast_avg_weekly"] < 0.5, "overstock_risk"] = np.where(
        df.loc[df["forecast_avg_weekly"] < 0.5, "on_hand_units"] > 5, 0.9, df.loc[df["forecast_avg_weekly"] < 0.5, "overstock_risk"])

    # --- Rupee value at stake ---
    df["sales_at_risk_rupees"] = (df["stockout_risk"] * demand_over_lead_time * df["list_price"]).round(0)
    excess_units = (df["on_hand_units"] - df["forecast_avg_weekly"] * OVERSTOCK_COVER_WEEKS).clip(lower=0)
    df["capital_locked_rupees"] = (excess_units * df["unit_cost"]).round(0)
    df["revenue_at_stake"] = df[["sales_at_risk_rupees", "capital_locked_rupees"]].max(axis=1)

    # --- Quadrant / recommended action (Section 08.2) ---
    def quadrant(row):
        so, ov = row["stockout_risk"] >= 0.5, row["overstock_risk"] >= 0.5
        if so and ov:
            return "Watch / Volatile", "Investigate — demand is erratic; review manually."
        if so:
            return "Reorder Now", "Raise a replenishment order before stock runs out."
        if ov:
            return "Markdown / Clear", "Promote or discount to free up capital."
        return "Healthy", "No action needed; leave as is."

    df[["risk_quadrant", "recommended_action"]] = df.apply(quadrant, axis=1, result_type="expand")

    cols = ["sku_id", "category", "subcategory", "forecast_avg_weekly", "forecast_horizon_units",
            "on_hand_units", "on_order_units", "lead_time_days", "reorder_point", "weeks_of_cover",
            "stockout_risk", "overstock_risk", "risk_quadrant", "recommended_action",
            "sales_at_risk_rupees", "capital_locked_rupees", "revenue_at_stake"]
    out = df[cols].sort_values("revenue_at_stake", ascending=False)
    out.to_csv(PROC / "risk_scored.csv", index=False)
    return out


if __name__ == "__main__":
    out = score()
    print(f"Scored {len(out)} SKUs.\n")
    print(out["risk_quadrant"].value_counts().to_string())
    print(f"\nTotal sales at risk (stockouts):  Rs {out['sales_at_risk_rupees'].sum():,.0f}")
    print(f"Total capital locked (overstock): Rs {out['capital_locked_rupees'].sum():,.0f}")
    print("\nTop 10 SKUs by revenue at stake:")
    print(out.head(10)[["sku_id", "category", "risk_quadrant", "revenue_at_stake"]].to_string(index=False))
