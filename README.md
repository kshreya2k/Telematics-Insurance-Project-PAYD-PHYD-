# Telematics-Insurance-Project-PAYD-PHYD-
A simple telematics-based auto insurance solution that accurately captures driving behavior and vehicle usage data and integrates this data into a dynamic insurance pricing model.

**Goal:** A tiny, runnable POC for usage-based auto insurance with synthetic telematics, simple features, baseline models, and a Streamlit UI. No SQL, no Docker.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (A) Generate synthetic trips + build features + train models from CLI
python -m src.data_gen.generate --rows 50000 --drivers 500
python -m src.etl.features --input data/trips.csv --out data/features.csv
python -m src.ml.train --features data/features.csv --models_dir models --metrics docs/metrics.json

# (B) Run the Streamlit app (can also generate/train from UI)
streamlit run app.py