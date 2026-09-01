# 3–5 Minute Demo Video — Talking Script

Record your screen (OBS / Loom / Zoom) walking through this in order. Aim for 4 minutes.

**0:00–0:30 — Frame the problem**
"NorthBay Living plans ~200 SKUs on spreadsheets. They stock out of best-sellers and
sit on slow movers. I built FORESIGHT to forecast demand and flag both risks
automatically."

**0:30–1:15 — Show the pipeline running (terminal)**
Run `bash run_all.sh` (or show it already run). Point out: "one command, from raw
CSVs to a cleaned, modelled dataset — no manual spreadsheet steps." Mention 1–2
real data-quality fixes from the log (e.g. "213 rows had missing revenue —
recomputed from units × price rather than dropped").

**1:15–2:00 — Backtest result**
Show `data/processed/backtest_results.csv` or the printed output. "Model WAPE 0.156
vs baseline 0.219 — beats seasonal-naive by 29%, on rolling-origin backtests the
model never trained on. This is the honest number, not an in-sample one."

**2:00–3:15 — Dashboard walkthrough**
Open the Streamlit app. Show: headline KPIs, the stockout/overstock decisioning
grid, filter to "Reorder Now", click into one SKU's forecast chart with the
uncertainty band. "This is what the ops team opens every Monday."

**3:15–3:45 — Scoring API**
Open `/docs` on the FastAPI service, call `GET /sku/{sku_id}` live, show the JSON
response. "Same forecast and risk, available as an API for any future tool."

**3:45–4:00 — Close**
"₹58.7L in sales at risk and ₹2.35L in locked capital identified this cycle — full
findings in the executive readout." Show the readout doc for 2 seconds. Done.
