import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_gen.generate import synth_trips
from src.etl.features import build_features
from src.ml.train import train_models, FEATURES
from src.ml.pricing import compute_premium, breakdown_row

st.set_page_config(page_title="Telematics Insurance POC", layout="wide")

st.title("Telematics Insurance (PAYD/PHYD) — Minimal POC")

# Sidebar nav
page = st.sidebar.radio("Navigation", [
    "Data: Upload / Generate",
    "Train Models",
    "Risk & Pricing Dashboard",
])

# Session state
if "trips" not in st.session_state:
    st.session_state.trips = None
if "features" not in st.session_state:
    st.session_state.features = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "metrics" not in st.session_state:
    st.session_state.metrics = {}

@st.cache_data(show_spinner=False)
def cache_build_features(df):
    return build_features(df)

@st.cache_resource(show_spinner=False)
def cache_train_models(features_df):
    return train_models(features_df)


# Page 1: Data

if page == "Data: Upload / Generate":
    st.subheader("1) Load trips data")

    c1, c2 = st.columns(2)
    with c1:
        uploaded = st.file_uploader(
            "Upload trips CSV",
            type=["csv"]
        )
        if uploaded is not None:
            trips = pd.read_csv(uploaded)
            st.session_state.trips = trips
            st.success(f"Loaded {len(trips):,} rows from upload")
            st.dataframe(trips.head(10))

    with c2:
        st.write("Or generate synthetic data:")
        n_rows = st.number_input("Rows", min_value=1000, max_value=500_000, value=50_000, step=5000)
        n_drivers = st.number_input("Drivers", min_value=50, max_value=10_000, value=500, step=50)
        if st.button("Generate synthetic trips"):
            trips = synth_trips(n_rows, n_drivers)
            st.session_state.trips = trips
            os.makedirs("data", exist_ok=True)
            trips.to_csv("data/trips.csv", index=False)
            st.success(f"Generated and saved to data/trips.csv ({len(trips):,} rows)")
            st.dataframe(trips.head(10))

    st.divider()
    if st.session_state.trips is not None:
        if st.button("Build features from trips"):
            feats = cache_build_features(st.session_state.trips)
            st.session_state.features = feats
            os.makedirs("data", exist_ok=True)
            feats.to_csv("data/features.csv", index=False)
            st.success(f"Built features for {len(feats):,} drivers → data/features.csv")
            st.dataframe(feats.head(10))


# Page 2: Training

elif page == "Train Models":
    st.subheader("2) Train Logistic Regression + Gradient Boosting")

    if st.session_state.features is None:
        if os.path.exists("data/features.csv"):
            st.session_state.features = pd.read_csv("data/features.csv")
        else:
            st.warning("No features available. Build them first in Data page.")

    if st.session_state.features is not None:
        with st.spinner("Training..."):
            lr, gbc, metrics = cache_train_models(st.session_state.features)
            st.session_state.models = {"logreg": lr, "gradboost": gbc}
            st.session_state.metrics = metrics

            os.makedirs("models", exist_ok=True)
            joblib.dump(lr, "models/logreg.joblib")
            joblib.dump(gbc, "models/gradboost.joblib")

            os.makedirs("docs", exist_ok=True)
            with open("docs/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        st.success("Models trained and saved.")
        st.json(metrics)



# Page 3: Dashboard

elif page == "Risk & Pricing Dashboard":
    st.subheader("3) Risk scoring and premiums")

    # show last metrics if available
    metrics_path = "docs/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            try:
                st.markdown("#### Last training metrics:")
                st.json(json.load(f))
            except Exception:
                pass

    if st.session_state.features is None and os.path.exists("data/features.csv"):
        st.session_state.features = pd.read_csv("data/features.csv")

    if st.session_state.features is None:
        st.warning("No features available. Build them first.")
    else:
        features_df = st.session_state.features.copy()

        if not st.session_state.models:
            try:
                st.session_state.models = {
                    "logreg": joblib.load("models/logreg.joblib"),
                    "gradboost": joblib.load("models/gradboost.joblib"),
                }
            except Exception:
                st.warning("No saved models found. Train models first.")

        if st.session_state.models:
            model_name = st.selectbox("Choose model", ["gradboost", "logreg"])
            model = st.session_state.models[model_name]

            base = st.number_input("Base premium ($)", min_value=200.0, max_value=5000.0, value=800.0, step=50.0)
            coverage_mult = st.slider("Coverage multiplier", 0.5, 2.0, 1.0, 0.05)
            deductible_mult = st.slider("Deductible multiplier", 0.7, 1.3, 1.0, 0.05)
            safe_thr = st.slider("Safe-driver threshold (risk)", 0.0, 0.5, 0.2, 0.01)
            safe_disc = st.slider("Safe-driver discount (%)", 0.0, 0.3, 0.1, 0.01)

            proba = model.predict_proba(features_df[FEATURES])[:, 1]
            features_df["risk_score"] = proba
            features_df["premium"] = compute_premium(
                features_df["risk_score"],
                base=base,
                coverage_multiplier=coverage_mult,
                deductible_multiplier=deductible_mult,
                safe_discount_threshold=safe_thr,
                safe_discount_pct=safe_disc,
            )

            st.markdown("### Driver Scores & Premiums")
            st.dataframe(
                features_df[["driver_id", "risk_score", "premium", "harsh_rate", "pct_night", "mean_speed", "accel_var"]]
                .sort_values("risk_score", ascending=False)
                .head(100)
            )

            st.markdown("### Visualizations")

            fig1 = plt.figure()
            plt.hist(features_df["risk_score"], bins=30)
            plt.title("Distribution of Risk Scores")
            st.pyplot(fig1)

            topn = st.slider("Top N risky drivers", 5, 100, 20)
            st.bar_chart(
                features_df.sort_values("risk_score", ascending=False)
                .head(topn)
                .set_index("driver_id")["risk_score"]
            )

            st.markdown("### Premium breakdown (single driver)")
            sel = st.selectbox("Driver", features_df["driver_id"].tolist())
            row = features_df.loc[features_df["driver_id"] == sel].iloc[0]
            raw, after_cov_ded, discount_amt, final, risk = breakdown_row(
                row["risk_score"], base, coverage_mult, deductible_mult, safe_thr, safe_disc
            )
            st.write({
                "driver_id": sel,
                "risk_score": float(risk),
                "base_x_(1+risk)": float(raw),
                "after_coverage_deductible": float(after_cov_ded),
                "safe_discount_amount": float(discount_amt),
                "final_premium": float(final),
            })

            if st.button("Export priced_table.csv"):
                out = features_df[["driver_id", "risk_score", "premium"]]
                out.to_csv("data/priced_table.csv", index=False)
                st.success("Saved to data/priced_table.csv")
