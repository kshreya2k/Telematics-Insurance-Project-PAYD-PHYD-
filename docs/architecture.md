# Telematics Insurance POC — Architecture

## 1. Purpose (What this POC shows)
A minimal, end-to-end pipeline that turns raw telematics-like trip data into:
1) driver-level **features**, 2) **risk scores** via ML models, and 3) **dynamic premiums** shown in a simple **Streamlit** dashboard.
Everything runs **locally** with Python 3.11, **no SQL and no Docker**.


## 2. Constraints & Tech Stack
- **Constraints:** Local only, CSV/Parquet storage, minimal dependencies.
- **Languages/Libs:** Python 3.11, Pandas, NumPy, scikit-learn, Streamlit, joblib, (optional) pyarrow for Parquet.
- **Persistence:** Flat files under `/data`, `/models`, `/docs`.


## 3. High-Level Design

```mermaid
flowchart LR
    A[Trips CSV (synthetic/uploaded)] --> B[ETL / Feature Builder]
    B --> C[Features CSV]
    C --> D[Model Training (LR + GBC)]
    D --> E[Risk Scoring]
    E --> F[Pricing Engine]
    A -->|via UI| G[Streamlit App]
    B -->|via UI| G
    C -->|metrics.json| G
    F -->|premiums| G