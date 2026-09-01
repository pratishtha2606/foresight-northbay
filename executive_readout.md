# FORESIGHT — Executive Readout
**For:** Head of Operations & Finance, NorthBay Living
**From:** Data Science & Analytics, Zidio Development
**Format:** 8-slide readout — copy each `##` section below into one slide.

---
## 1. The bottom line
- **₹58.7 lakh–crore-scale of sales are currently at risk** across 34 SKUs projected to stock out within their replenishment lead time.
- **₹2.35 lakh of capital is sitting in slow-moving stock** that should be marked down or cleared.
- A forecasting model built on your own sales history **beats a same-period-last-year baseline by ~29%** on out-of-sample backtests — it's ready to drive real reorder decisions.

*(Figures are computed on the current data snapshot from `risk_scored.csv` — refresh monthly by re-running the pipeline.)*

---
## 2. The problem, in one sentence
NorthBay plans ~200 SKUs on gut feel and spreadsheets; some sell out while others pile up, costing lost sales and locked cash in both directions.

---
## 3. What we built
1. A reproducible pipeline that cleans and unifies sales, inventory, calendar, and product data.
2. A weekly, SKU-level demand forecast, backtested honestly against a naive baseline.
3. A stockout/overstock risk score for every SKU, with a recommended action and rupee value attached.
4. A planning dashboard the ops team can use without a data scientist in the room.
5. A live scoring API so the forecast and risk can be pulled into any future tool.

---
## 4. Forecast accuracy — honestly reported
| | WAPE (lower = better) |
|---|---|
| Seasonal-naive baseline | ~22% |
| FORESIGHT model | ~16% |

Backtested with rolling-origin cross-validation (train on the past, test on the next 6 weeks, repeated 4 times) — never trained and tested on the same period. This is the accuracy you should expect going forward, not an optimistic in-sample number.

**Limitations, stated plainly:** brand-new SKUs with little history fall back to a simpler recent-average forecast until they build up sales data. Forecasts assume future promotions look like past promotions — a materially different promo calendar will need a manual override.

---
## 5. Where to act first
The top 10 SKUs by revenue at stake concentrate in **Small Appliances, Decor, and Furniture** — reordering these first captures the majority of the risk currently on the books. See the "Reorder Now" list in the dashboard for the full, prioritised set.

---
## 6. Seasonality — plan ahead for Nov–Dec
Weekly demand nearly doubles in November–December versus the rest of the year (Diwali and Year-End promotions). Lead times should be padded and reorders placed early heading into that window — a stockout during peak promotion is the costliest kind.

---
## 7. What NorthBay's team does with this monthly
1. Open the dashboard, filter to "Reorder Now" — raise POs for those SKUs first.
2. Filter to "Markdown / Clear" — hand that list to merchandising for promotion/clearance.
3. Re-run the pipeline against the latest data export to refresh both lists.

---
## 8. Next steps we recommend
- Wire the pipeline to the client's live sales/inventory export (currently runs on point-in-time extracts — out of scope for this engagement, in scope for a phase 2).
- Add a lightweight model-monitoring check (compare next month's actuals to this month's forecast) so degradation is caught early.
- Revisit safety-stock assumptions with Finance once 2–3 months of live results are in.
