# Project FORESIGHT — Data-Quality & EDA Insight Memo

**Prepared for:** Head of Operations, NorthBay Living
**Prepared by:** Data Science & Analytics, Zidio Development
**Data window:** Sep 2023 – Aug 2025 (≈2 years, 200 SKUs, daily sales)

## 1. Data-quality issues found and how they were handled

| Issue | Found in | How it was handled |
|---|---|---|
| Inconsistent category labels ("Décor" / "decor" / "Decor") | `sku_master` | Normalised to a canonical set of 4 categories before any aggregation. |
| Duplicate SKU rows | `sku_master` | 5 duplicate rows dropped, keeping first occurrence. |
| Missing `unit_cost` | `sku_master` | 6 rows — imputed with the **category median cost**, not the global median, since cost structure varies sharply by category. |
| Missing `revenue` | `sales_daily` | 213 rows — recomputed as `units_sold × unit_price` rather than dropped, since revenue is fully derivable from other columns in the same row. |
| Duplicate transaction rows | `sales_daily` | 50 exact duplicate rows dropped. |
| Sparse history for new SKUs | `sales_daily` | SKUs with under ~10 weeks of history are excluded from backtesting and fall back to a simple recent-average forecast rather than the full model (see Section 16.1 of the brief — sparse-history risk). |

All cleaning is coded in `src/pipeline.py` and re-runs end-to-end from the raw extracts; nothing was cleaned by hand in a spreadsheet.

## 2. Demand patterns

**Top movers (units sold, full history):** SKU0177, SKU0138, SKU0061, SKU0075, SKU0180 lead the catalogue, each moving 25,000+ units over two years — these are the SKUs a stockout hurts most.

**Dead-stock candidates:** SKU0135, SKU0151, SKU0030, SKU0038, SKU0080 sold under 450 units combined over the same two-year window — candidates for markdown/clearance review regardless of what the risk model says, simply on velocity.

**Demand by category (units, full history):**
Furniture (495k) > Decor (456k) > Textiles (371k) > Small Appliances (300k). Furniture and Decor together account for over half of unit demand — inventory planning attention should weight toward these two categories.

**Seasonality:** average weekly units climb from ~145–150 in most months to **186 in November** and **198 in December**, consistent with the Diwali and Year-End promotional windows in the calendar data. January–October is comparatively flat. This is the seasonal pattern the seasonal-naive baseline and the model's lag-52 feature are built to capture.

**Promotion effect:** weeks with a promotion active average **219 units/week** vs **148 units/week** without — a ~48% lift. Promotion flags are a material forecast driver and are included as a model feature.

## 3. Business-relevant insights (plain language)

1. **A small set of SKUs drive a disproportionate share of stockout risk.** The risk-scoring output concentrates sales-at-risk in ~30 SKUs (see `risk_scored.csv`), mostly in Small Appliances, Decor, and Furniture — this is where the ops team should focus reorder attention first, not spread effort evenly across all 200 SKUs.
2. **November–December is the highest-stakes stocking window.** Demand nearly doubles versus the January–October baseline, so lead-time buffers and reorder timing matter most heading into that period — running out during Diwali/Year-End promotions is the costliest possible stockout.
3. **A long tail of slow movers is quietly locking up capital.** Several dozen SKUs sell in the low hundreds of units over two years while still carrying on-hand stock — these are markdown/clear candidates independent of any forecasting sophistication, and are a quick win the ops team can act on immediately.

## 4. What this means for the model

The clear seasonality and promotion lift justify including calendar and promo features rather than relying on a naive trend-only forecast, and justify comparing against a **seasonal-naive baseline** (same week, prior year) rather than a flat-average baseline — see `src/forecast.py` and the backtest results in the README.
