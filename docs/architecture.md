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
    A[Trips CSV (synthetic or uploaded)] --> B[ETL / Feature Builder]
    B -->|features.csv| C[Model Training (LR + GBC)]
    C -->|joblib| D[Risk Scoring]
    D --> E[Pricing Engine]
    A -->|via UI| UI[Streamlit App]
    B -->|via UI| UI
    C -->|metrics.json| UI
    E -->|premiums table| UI
```

## 4. Limitations & Future Work: Cloud and Scalability
This POC is intentionally lightweight — it runs locally using CSV/Parquet files, Pandas, and Streamlit.  
In a production deployment, the following enhancements would be required:

- **Scalable Cloud Infrastructure**  
  Use managed cloud services (e.g., AWS S3/GCP Cloud Storage for raw telematics data, BigQuery/Redshift/Snowflake for feature stores, and Spark or Dataflow for large-scale ETL).
- **Streaming Ingestion**  
  Real-time data from telematics devices would be ingested via secure APIs or message queues (Kafka, Pub/Sub, Kinesis).
- **Model Serving**  
  Risk scoring models would be containerized (e.g., with Docker) and deployed on Kubernetes, Vertex AI, or SageMaker for scalable inference.
- **Security & Compliance**  
  Enforce encryption, role-based access control, and anonymization/pseudonymization for sensitive driving and location data.
- **Integration**  
  APIs would expose driver scores and premium calculations to insurance portals and mobile apps.

*By keeping this POC local and minimal, we demonstrate the end-to-end workflow clearly. Future iterations can migrate these components into a secure, scalable cloud environment without changing the core logic.*
