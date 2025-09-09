# Architecture (POC)

```mermaid
flowchart LR
A[Generator / Upload] --> B[ETL/Feature Builder]
B --> C[Models: LR + GBC]
C --> D[Risk Score]
D --> E[Pricing]
A -->|CSV/Parquet| B
C -->|joblib| UI[Streamlit UI]
B -->|features.csv| UI
UI -->|metrics.json| Docs


[Trips CSV] -> [Features CSV] -> [Train LR/GBC] -> [Risk Score] -> [Pricing]
\_____________________________________________/ |
Streamlit UI reads/writes <--------------