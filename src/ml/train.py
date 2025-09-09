# src/ml/train.py
import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURES = [
    "total_miles", "records", "mean_speed", "std_speed", "accel_mean", "accel_var",
    "harsh_rate", "pct_night", "road_type_highway", "road_type_rural", "road_type_urban",
]
TARGET = "had_claim"


def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression pipeline (with scaling)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

    # Gradient Boosting baseline
    gbc = GradientBoostingClassifier(random_state=42)

    lr.fit(X_train, y_train)
    gbc.fit(X_train, y_train)

    # Evaluation
    preds_lr = lr.predict(X_test)
    proba_lr = lr.predict_proba(X_test)[:, 1]
    preds_gb = gbc.predict(X_test)
    proba_gb = gbc.predict_proba(X_test)[:, 1]

    metrics = {
        "logreg": {
            "accuracy": float(accuracy_score(y_test, preds_lr)),
            "auc": float(roc_auc_score(y_test, proba_lr)),
        },
        "gradboost": {
            "accuracy": float(accuracy_score(y_test, preds_gb)),
            "auc": float(roc_auc_score(y_test, proba_gb)),
        }
    }

    return lr, gbc, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=os.path.join("data", "features.csv"))
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--metrics", type=str, default=os.path.join("docs", "metrics.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)

    lr, gbc, metrics = train_models(df)

    # Save models
    joblib.dump(lr, os.path.join(args.models_dir, "logreg.joblib"))
    joblib.dump(gbc, os.path.join(args.models_dir, "gradboost.joblib"))

    # Save metrics
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved models to:")
    print(" -", os.path.join(args.models_dir, "logreg.joblib"))
    print(" -", os.path.join(args.models_dir, "gradboost.joblib"))
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
