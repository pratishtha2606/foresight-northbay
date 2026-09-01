"""
Project FORESIGHT — D6 Deployed scoring service.

Run locally:
    uvicorn service.main:app --reload --port 8000
    Then open http://localhost:8000/docs for interactive API docs.

Deploy free on Render.com:
    1. Push repo to GitHub.
    2. New Web Service on Render, connect the repo.
    3. Start command: uvicorn service.main:app --host 0.0.0.0 --port $PORT
    4. Render gives you a public URL — that's your D6 submission link.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

app = FastAPI(
    title="FORESIGHT Scoring Service",
    description="Returns demand forecast + stockout/overstock risk for a NorthBay Living SKU.",
    version="1.0.0",
)

_risk = None
_forecasts = None


def _load():
    global _risk, _forecasts
    if _risk is None:
        try:
            _risk = pd.read_csv(PROC / "risk_scored.csv").set_index("sku_id")
            _forecasts = pd.read_csv(PROC / "forecasts.csv", parse_dates=["week_start"])
        except FileNotFoundError as e:
            raise HTTPException(status_code=503,
                                 detail="Scoring data not built yet. Run the pipeline first.") from e
    return _risk, _forecasts


class BatchRequest(BaseModel):
    sku_ids: list[str]


@app.get("/")
def root():
    return {"service": "FORESIGHT scoring", "status": "ok",
            "endpoints": ["/sku/{sku_id}", "/batch (POST)", "/health"]}


@app.get("/health")
def health():
    try:
        risk, _ = _load()
        return {"status": "ok", "skus_available": len(risk)}
    except HTTPException:
        return {"status": "data_not_built"}


@app.get("/sku/{sku_id}")
def get_sku(sku_id: str):
    risk, forecasts = _load()
    sku_id = sku_id.strip().upper()
    if sku_id not in risk.index:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found.")
    r = risk.loc[sku_id]
    fc = forecasts[forecasts["sku_id"] == sku_id][
        ["week_start", "forecast_units", "forecast_lower", "forecast_upper"]
    ].copy()
    fc["week_start"] = fc["week_start"].astype(str)
    return {
        "sku_id": sku_id,
        "category": r["category"],
        "risk_quadrant": r["risk_quadrant"],
        "recommended_action": r["recommended_action"],
        "stockout_risk": round(float(r["stockout_risk"]), 3),
        "overstock_risk": round(float(r["overstock_risk"]), 3),
        "weeks_of_cover": round(float(r["weeks_of_cover"]), 1),
        "revenue_at_stake_rupees": float(r["revenue_at_stake"]),
        "forecast": fc.to_dict(orient="records"),
    }


@app.post("/batch")
def get_batch(req: BatchRequest):
    risk, forecasts = _load()
    results = []
    for sku_id in req.sku_ids:
        sku_id = sku_id.strip().upper()
        if sku_id not in risk.index:
            results.append({"sku_id": sku_id, "error": "not found"})
            continue
        r = risk.loc[sku_id]
        results.append({
            "sku_id": sku_id,
            "risk_quadrant": r["risk_quadrant"],
            "recommended_action": r["recommended_action"],
            "stockout_risk": round(float(r["stockout_risk"]), 3),
            "overstock_risk": round(float(r["overstock_risk"]), 3),
            "revenue_at_stake_rupees": float(r["revenue_at_stake"]),
        })
    return {"results": results}
