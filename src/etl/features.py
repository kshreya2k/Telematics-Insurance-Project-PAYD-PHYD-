# src/etl/features.py
import argparse
import os
import pandas as pd
import numpy as np

def build_features(trips: pd.DataFrame) -> pd.DataFrame:
    df = trips.copy()

    # Basic cleaning
    need_cols = ["driver_id", "speed_mph", "accel_ms2", "mileage", "harsh_brake", "is_night"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["harsh_brake"] = df["harsh_brake"].astype(int)
    df["is_night"] = df["is_night"].astype(int)
    if "is_peak" not in df.columns:
        df["is_peak"] = 0  # backward compatible if old trips.csv is used
    df["is_peak"] = df["is_peak"].astype(int)

    # Aggregate per driver
    agg = df.groupby("driver_id").agg(
        total_miles=("mileage", "sum"),
        records=("mileage", "count"),
        mean_speed=("speed_mph", "mean"),
        std_speed=("speed_mph", "std"),
        accel_mean=("accel_ms2", "mean"),
        accel_var=("accel_ms2", np.var),  # OK with pandas FutureWarning
        harsh_count=("harsh_brake", "sum"),
        night_count=("is_night", "sum"),
        peak_count=("is_peak", "sum"),
    ).reset_index()

    # Rates & shares
    agg["harsh_rate"] = agg["harsh_count"] / agg["records"].clip(lower=1)
    agg["pct_night"] = agg["night_count"] / agg["records"].clip(lower=1)
    agg["pct_peak"] = agg["peak_count"] / agg["records"].clip(lower=1)

    # Road type distribution
    if "road_type" in df.columns:
        road_dummies = pd.get_dummies(df[["driver_id", "road_type"]], columns=["road_type"]).groupby("driver_id").sum()
        road_share = road_dummies.div(road_dummies.sum(axis=1), axis=0).reset_index()
    else:
        road_share = pd.DataFrame({"driver_id": agg["driver_id"]})  # empty; will fill zeros below

    feats = agg.merge(road_share, on="driver_id", how="left")

    # Synthetic target: claim probability
    risk_logit = (
        3.0 * feats["harsh_rate"].fillna(0)
        + 2.0 * feats["pct_night"].fillna(0)
        + 1.0 * feats["pct_peak"].fillna(0)
        + 0.8 * feats["accel_var"].fillna(0)
        - 0.01 * feats["mean_speed"].fillna(feats["mean_speed"].median())
        + 0.2 * (1 - feats.get("road_type_highway", 0).fillna(0))
    )
    prob = 1 / (1 + np.exp(-risk_logit))
    rng = np.random.default_rng(123)
    feats["had_claim"] = (rng.random(len(feats)) < prob.clip(0, 1)).astype(int)

    # Ensure road type columns exist
    for c in ["road_type_highway", "road_type_rural", "road_type_urban"]:
        if c not in feats.columns:
            feats[c] = 0.0

    cols_num = [
        "total_miles", "records", "mean_speed", "std_speed", "accel_mean", "accel_var",
        "harsh_rate", "pct_night", "pct_peak",
        "road_type_highway", "road_type_rural", "road_type_urban",
    ]
    return feats[["driver_id", "had_claim"] + cols_num]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join("data", "trips.csv"))
    parser.add_argument("--out", type=str, default=os.path.join("data", "features.csv"))
    args = parser.parse_args()

    trips = pd.read_csv(args.input)
    feats = build_features(trips)

    os.makedirs("data", exist_ok=True)
    feats.to_csv(args.out, index=False)
    try:
        feats.to_parquet(args.out.replace(".csv", ".parquet"), index=False)
    except Exception:
        pass

    print(f"Wrote features: {args.out} with {len(feats):,} drivers")

if __name__ == "__main__":
    main()
