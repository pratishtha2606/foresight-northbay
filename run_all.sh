#!/usr/bin/env bash
# Project FORESIGHT — reproduce everything from raw data with one command.
set -e
echo "== 1/4  Generating synthetic client data (skip if data/raw already populated) =="
python3 src/generate_data.py

echo "== 2/4  Cleaning pipeline =="
python3 src/pipeline.py

echo "== 3/4  Backtest + forecast =="
python3 src/forecast.py

echo "== 4/4  Risk scoring =="
python3 src/risk.py

echo ""
echo "Done. Now run:"
echo "  streamlit run app/dashboard.py        # planning dashboard"
echo "  uvicorn service.main:app --reload      # scoring API on http://localhost:8000/docs"
