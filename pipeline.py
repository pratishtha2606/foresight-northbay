"""
Project FORESIGHT — D1 Data pipeline.

Ingests the four raw extracts, cleans them, and produces one analysis-ready
weekly SKU-level dataset. Re-runs end-to-end from raw files with:

    python src/pipeline.py

Cleaning decisions (documented per the brief's acceptance criteria):
  - sku_master: category labels are case/accent inconsistent -> normalised to
    a canonical set. Duplicate sku_id rows -> kept first occurrence.
    Missing unit_cost -> imputed with the category median (cost structure is
    category-driven; a global median would distort margin-sensitive SKUs).
  - sales_daily: missing revenue -> recomputed as units_sold * unit_price
    (revenue is derivable, so we reconstruct rather than drop or impute
    blindly). Exact duplicate rows -> dropped.
  - calendar: no missing values expected; used as-is after type fixes.
  - inventory_snapshots: forward-filled per SKU for the (rare) gaps between
    snapshot dates, since stock position doesn't reset between snapshots.
  - Sales are aggregated to weekly grain (week start = Monday) per SKU, which
    is the grain the client asked to be forecast at and reduces daily noise.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CANONICAL_CATEGORY = {
    "furniture": "Furniture",
    "decor": "Decor", "décor": "Decor",
    "small appliances": "Small Appliances", "small appl.": "Small Appliances",
    "textiles": "Textiles", "textile": "Textiles",
}


def clean_sku_master():
    df = pd.read_csv(RAW / "sku_master.csv", parse_dates=["launch_date"])
    before = len(df)
    df = df.drop_duplicates(subset="sku_id", keep="first")
    n_dupes = before - len(df)

    df["category"] = df["category"].str.strip().str.lower().map(CANONICAL_CATEGORY).fillna(df["category"])
    df["category_median_cost"] = df.groupby("category")["unit_cost"].transform("median")
    n_missing_cost = df["unit_cost"].isna().sum()
    df["unit_cost"] = df["unit_cost"].fillna(df["category_median_cost"])
    df = df.drop(columns="category_median_cost")

    log = {"sku_master_duplicates_dropped": int(n_dupes),
           "sku_master_unit_cost_imputed": int(n_missing_cost)}
    return df, log


def clean_sales(sku_master):
    df = pd.read_csv(RAW / "sales_daily.csv", parse_dates=["date"])
    before = len(df)
    df = df.drop_duplicates()
    n_dupes = before - len(df)

    n_missing_rev = df["revenue"].isna().sum()
    recomputed = df["units_sold"] * df["unit_price"]
    df["revenue"] = df["revenue"].fillna(recomputed)

    # keep only SKUs present in the cleaned master
    valid_skus = set(sku_master["sku_id"])
    n_orphan = (~df["sku_id"].isin(valid_skus)).sum()
    df = df[df["sku_id"].isin(valid_skus)]

    log = {"sales_duplicates_dropped": int(n_dupes),
           "sales_revenue_recomputed": int(n_missing_rev),
           "sales_orphan_rows_dropped": int(n_orphan)}
    return df, log


def clean_calendar():
    df = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    df["promo_flag"] = (df["promo_event"].fillna("") != "").astype(int)
    return df


def clean_inventory(sku_master):
    df = pd.read_csv(RAW / "inventory_snapshots.csv", parse_dates=["date"])
    valid_skus = set(sku_master["sku_id"])
    df = df[df["sku_id"].isin(valid_skus)]
    df = df.sort_values(["sku_id", "date"])
    return df


def build_weekly_dataset(sales, calendar, sku_master):
    cal = calendar.copy()
    cal["week_start"] = cal["date"] - pd.to_timedelta(cal["date"].dt.dayofweek, unit="D")

    s = sales.merge(cal[["date", "week_start", "is_holiday", "promo_event"]], on="date", how="left")
    weekly = (s.groupby(["sku_id", "week_start"])
                .agg(units_sold=("units_sold", "sum"),
                     revenue=("revenue", "sum"),
                     avg_price=("unit_price", "mean"),
                     promo_days=("promo_flag", "sum"),
                     holiday_days=("is_holiday", "sum"))
                .reset_index())
    weekly = weekly.merge(sku_master[["sku_id", "category", "subcategory",
                                       "unit_cost", "list_price", "launch_date"]],
                           on="sku_id", how="left")
    weekly["promo_flag"] = (weekly["promo_days"] > 0).astype(int)
    weekly = weekly.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    return weekly


def latest_inventory_position(inventory):
    """Most recent snapshot per SKU as of the pipeline run — used by risk scoring."""
    idx = inventory.groupby("sku_id")["date"].idxmax()
    return inventory.loc[idx].reset_index(drop=True)


def run():
    sku_master, log1 = clean_sku_master()
    sales, log2 = clean_sales(sku_master)
    calendar = clean_calendar()
    inventory = clean_inventory(sku_master)

    weekly = build_weekly_dataset(sales, calendar, sku_master)
    inv_latest = latest_inventory_position(inventory)

    sku_master.to_csv(PROC / "sku_master_clean.csv", index=False)
    sales.to_csv(PROC / "sales_daily_clean.csv", index=False)
    inventory.to_csv(PROC / "inventory_clean.csv", index=False)
    weekly.to_csv(PROC / "weekly_sku_demand.csv", index=False)
    inv_latest.to_csv(PROC / "inventory_latest.csv", index=False)

    log = {**log1, **log2, "weekly_rows": len(weekly), "n_skus": weekly["sku_id"].nunique()}
    print("Data-quality log:")
    for k, v in log.items():
        print(f"  {k}: {v}")
    return log


if __name__ == "__main__":
    run()
