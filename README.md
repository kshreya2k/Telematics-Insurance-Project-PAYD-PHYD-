# Telematics-Insurance-Project-PAYD-PHYD-
A simple Proof of Concept(POC) for telematics-based auto insurance solution that accurately captures driving behavior and vehicle usage data and integrates this data into a dynamic insurance pricing model. 

**Goal:** A simple, clean and runnable POC for usage-based auto insurance with synthetic telematics, simple features, baseline models, and a Streamlit UI.

## Features
- **Synthetic Data Generator**: Produces trip-level telematics data (timestamp, speed, accel, harsh braking, mileage, night flag, peak-hour flag, road type, GPS coords).  
- **ETL / Feature Engineering**: Aggregates per-driver features like harsh-brake rate, % night driving, % peak-hour, speed variability, and road type distribution.  
- **Risk Models**: Logistic Regression and Gradient Boosting Classifier trained on synthetic claim labels. Metrics: Accuracy + AUC.  
- **Pricing Engine**: Premium = `base × (1 + risk_score)` with coverage/deductible multipliers and safe-driver discounts.  
- **Streamlit Dashboard**:  
  - Upload or generate trip data  
  - Train models and view evaluation metrics  
  - Explore driver scores, premiums, and charts  
  - Export priced table (`data/priced_table.csv`)  

## Machine Learning Models & Tools

This POC uses **two baseline models** for driver risk scoring:  

- **Logistic Regression**  
  - Simple, interpretable model.  
  - Well-suited for binary classification problems (e.g., claim vs. no claim).  
  - Easy to explain to business stakeholders.  

- **Gradient Boosting Classifier**  
  - A tree-based ensemble method that can capture non-linear patterns in driving behavior.  
  - Provides a stronger baseline compared to linear models.  

The intention is to keep the models **simple and lightweight**:  
- Fast to train and run on synthetic datasets.  
- Avoids overfitting, which is common with small or simulated data.  
- Clear results (accuracy and AUC) for easy comparison.  

## Tools Used
- **Python 3.11** with `venv`: minimal, reproducible environment.  
- **Pandas & NumPy**: for data cleaning, feature engineering, and aggregation.  
- **scikit-learn**: for model training, evaluation, and joblib persistence.  
- **Streamlit**: quick and interactive dashboard to explore data, train models, and visualize premiums.  
- **CSV/Parquet storage**: no database, keeps things simple and portable.  

*This design choice keeps the POC **easy to run, understand, and extend**, while still proving the core concept of usage-based insurance with telematics data.*

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
