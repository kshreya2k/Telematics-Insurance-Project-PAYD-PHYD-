# Telematics Insurance POC — Documentation

## 1. Introduction
This project is a **proof of concept (POC)** for a telematics-based auto insurance solution.  
It demonstrates how **driver behavior data** (speed, braking, mileage, time-of-travel, GPS) can be used to build a **risk scoring model** and calculate **dynamic insurance premiums**.  

The POC is intentionally lightweight: it uses Python 3.11, Pandas, scikit-learn, and Streamlit, with CSV/Parquet files for storage.  

## 2. Objectives
- Capture and process telematics-like trip data (synthetic for this POC).  
- Engineer per-driver features related to risky behaviors.  
- Train baseline ML models to predict claim probability.  
- Integrate a simple pricing engine based on risk scores.  
- Provide a transparent web dashboard (Streamlit) for data exploration, training, and premium visualization.  


## 3. System Architecture

### High-Level Flow
1. **Data Generation** → synthetic telematics trips (`trips.csv`)  
2. **ETL & Features** → aggregate per-driver metrics (`features.csv`)  
3. **Model Training** → Logistic Regression + Gradient Boosting, save models & metrics  
4. **Pricing Engine** → compute dynamic premiums based on risk score  
5. **Streamlit Dashboard** → end-to-end UI for users to explore  

```mermaid
flowchart LR
    A[Trips (synthetic/uploaded CSV)] --> B[ETL / Features]
    B --> C[Model Training]
    C --> D[Risk Scoring]
    D --> E[Pricing Engine]
    E --> F[Streamlit Dashboard]