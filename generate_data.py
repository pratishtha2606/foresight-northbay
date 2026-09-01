"""
Project FORESIGHT — synthetic data generator.
Produces the four raw extracts described in the client brief (Appendix A):
sales_daily, sku_master, calendar, inventory_snapshots.

Data is deliberately imperfect: missing values, a few duplicates, and
inconsistent category labels, so the pipeline has real cleaning to do.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

N_SKUS = 200
START = pd.Timestamp("2023-09-01")
END = pd.Timestamp("2025-08-31")  # ~2 years of daily history
DATES = pd.date_range(START, END, freq="D")

CATEGORIES = {
    "Furniture": ["Chairs", "Tables", "Storage"],
    "Decor": ["Wall Art", "Lighting", "Rugs"],
    "Small Appliances": ["Kitchen", "Climate", "Cleaning"],
    "Textiles": ["Bedding", "Cushions", "Throws"],
}
# Inconsistent label variants injected on purpose (data-quality issue to clean)
CATEGORY_VARIANTS = {
    "Furniture": ["Furniture", "furniture", "FURNITURE"],
    "Decor": ["Decor", "Décor", "decor"],
    "Small Appliances": ["Small Appliances", "small appliances", "Small Appl."],
    "Textiles": ["Textiles", "textiles", "Textile"],
}


def make_sku_master():
    rows = []
    cats = list(CATEGORIES.keys())
    for i in range(1, N_SKUS + 1):
        sku_id = f"SKU{i:04d}"
        cat = RNG.choice(cats, p=[0.3, 0.3, 0.2, 0.2])
        subcat = RNG.choice(CATEGORIES[cat])
        launch_offset = RNG.integers(0, (END - START).days - 30)
        launch_date = START + pd.Timedelta(days=int(launch_offset))
        unit_cost = round(RNG.uniform(150, 4000), 2)
        margin = RNG.uniform(1.4, 2.6)
        list_price = round(unit_cost * margin, 2)
        cat_label = RNG.choice(CATEGORY_VARIANTS[cat])  # inject label inconsistency
        rows.append([sku_id, cat_label, subcat, launch_date, unit_cost, list_price])
    df = pd.DataFrame(rows, columns=["sku_id", "category", "subcategory",
                                      "launch_date", "unit_cost", "list_price"])
    # inject a few duplicate rows and missing unit_cost
    dupes = df.sample(5, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)
    miss_idx = df.sample(6, random_state=2).index
    df.loc[miss_idx, "unit_cost"] = np.nan
    return df


def make_calendar():
    df = pd.DataFrame({"date": DATES})
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["season"] = df["month"] % 12 // 3 + 1
    df["season"] = df["season"].map({1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"})
    # simple holiday calendar (India-ish major sale/holiday dates, illustrative)
    holiday_dates = set()
    for y in range(START.year, END.year + 1):
        holiday_dates.update([
            pd.Timestamp(y, 1, 1), pd.Timestamp(y, 8, 15), pd.Timestamp(y, 10, 2),
            pd.Timestamp(y, 12, 25),
        ])
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    df["promo_event"] = ""
    promo_windows = [
        ("2023-11-20", "2023-11-27", "Diwali Sale"),
        ("2023-12-20", "2023-12-31", "Year End Sale"),
        ("2024-06-01", "2024-06-10", "Summer Clearance"),
        ("2024-11-08", "2024-11-15", "Diwali Sale"),
        ("2024-12-20", "2024-12-31", "Year End Sale"),
        ("2025-06-01", "2025-06-10", "Summer Clearance"),
    ]
    for s, e, name in promo_windows:
        mask = (df["date"] >= s) & (df["date"] <= e)
        df.loc[mask, "promo_event"] = name
    return df


def make_sales(sku_master, calendar):
    cal = calendar.set_index("date")
    all_rows = []
    for _, sku in sku_master.drop_duplicates("sku_id").iterrows():
        sku_id = sku["sku_id"]
        launch = sku["launch_date"]
        base = RNG.uniform(2, 40)  # base daily demand level
        trend = RNG.uniform(-0.0005, 0.0008)
        weekly_amp = RNG.uniform(0.1, 0.4)
        season_amp = RNG.uniform(0.05, 0.35)
        noise_sigma = RNG.uniform(0.2, 0.5)
        dates = DATES[DATES >= launch]
        if len(dates) == 0:
            continue
        t = np.arange(len(dates))
        dow = dates.dayofweek.values
        doy = dates.dayofyear.values
        weekly = 1 + weekly_amp * np.sin(2 * np.pi * dow / 7)
        seasonal = 1 + season_amp * np.sin(2 * np.pi * doy / 365 + RNG.uniform(0, 2 * np.pi))
        trend_component = 1 + trend * t
        promo_flag = cal.loc[dates, "promo_event"].astype(bool).values.astype(int)
        promo_lift = 1 + 0.9 * promo_flag
        holiday_lift = 1 + 0.3 * cal.loc[dates, "is_holiday"].values
        lam = np.clip(base * weekly * seasonal * trend_component * promo_lift * holiday_lift, 0.05, None)
        units = RNG.poisson(lam) 
        units = np.round(units * (1 + RNG.normal(0, noise_sigma, size=len(units)))).clip(min=0).astype(int)
        price = sku["list_price"]
        price_series = np.where(promo_flag == 1, price * RNG.uniform(0.75, 0.9), price)
        revenue = units * price_series
        all_rows.append(pd.DataFrame({
            "date": dates, "sku_id": sku_id, "units_sold": units,
            "revenue": np.round(revenue, 2), "unit_price": np.round(price_series, 2),
            "promo_flag": promo_flag,
        }))
    df = pd.concat(all_rows, ignore_index=True)
    # inject missing values & a duplicate block (data-quality issue)
    miss_idx = df.sample(frac=0.003, random_state=3).index
    df.loc[miss_idx, "revenue"] = np.nan
    dupe_block = df.sample(50, random_state=4)
    df = pd.concat([df, dupe_block], ignore_index=True)
    return df


def make_inventory(sku_master, sales):
    rows = []
    for _, sku in sku_master.drop_duplicates("sku_id").iterrows():
        sku_id = sku["sku_id"]
        sub = sales[sales["sku_id"] == sku_id]
        if sub.empty:
            continue
        avg_daily = max(sub["units_sold"].mean(), 0.5)
        lead_time = int(RNG.integers(5, 21))
        reorder_point = round(avg_daily * lead_time * RNG.uniform(1.0, 1.5))
        snap_dates = pd.date_range(sub["date"].min(), END, freq="7D")
        on_hand = max(avg_daily * RNG.uniform(10, 40), 5)
        for d in snap_dates:
            recent_sales = sub[(sub["date"] >= d - pd.Timedelta(days=7)) & (sub["date"] < d)]["units_sold"].sum()
            on_hand = max(on_hand - recent_sales + RNG.poisson(avg_daily * 7 * RNG.uniform(0.6, 1.3)), 0)
            on_order = RNG.poisson(avg_daily * lead_time * 0.3)
            rows.append([d, sku_id, int(round(on_hand)), int(on_order), lead_time, int(reorder_point)])
    df = pd.DataFrame(rows, columns=["date", "sku_id", "on_hand_units", "on_order_units",
                                      "lead_time_days", "reorder_point"])
    return df


if __name__ == "__main__":
    sku_master = make_sku_master()
    calendar = make_calendar()
    sales = make_sales(sku_master, calendar)
    inventory = make_inventory(sku_master, sales)

    sku_master.to_csv(RAW / "sku_master.csv", index=False)
    calendar.to_csv(RAW / "calendar.csv", index=False)
    sales.to_csv(RAW / "sales_daily.csv", index=False)
    inventory.to_csv(RAW / "inventory_snapshots.csv", index=False)

    print("Generated:")
    print(" sku_master        ", sku_master.shape)
    print(" calendar          ", calendar.shape)
    print(" sales_daily       ", sales.shape)
    print(" inventory_snapshots", inventory.shape)
