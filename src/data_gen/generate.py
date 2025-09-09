import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproducibility
RNG = np.random.default_rng(42)

ROAD_TYPES = ["highway", "urban", "rural"]

def synth_trips(n_rows: int = 50_000, n_drivers: int = 500) -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    driver_ids = [f"D{str(i).zfill(5)}" for i in range(n_drivers)]

    # Random driver risk profiles to bias behavior
    driver_risk = pd.DataFrame({
    "driver_id": driver_ids,
    "risk_bias": RNG.uniform(0, 1, size=n_drivers), # higher => riskier behavior
    })

    rows = []

    for _ in range(n_rows):
        d = driver_risk.sample(1, weights=None, random_state=None).iloc[0]
        driver_id = d.driver_id
        rb = d.risk_bias

        # Random timestamp over ~180 days
        ts = start + timedelta(minutes=int(RNG.integers(0, 180*24*60)))
        hour = ts.hour
        is_night = int(hour < 6 or hour >= 22)

        road = RNG.choice(ROAD_TYPES, p=[0.45, 0.4, 0.15])
        base_speed = {
            "highway": RNG.normal(65, 7),
            "urban": RNG.normal(30, 6),
            "rural": RNG.normal(45, 8),
        }[road]
        speed = max(0, base_speed + RNG.normal(0, 5) + rb * 10 * RNG.uniform(-0.2, 1.0))

        # Acceleration (m/s^2) approx — higher variance for riskier drivers
        accel = RNG.normal(0.8 + rb * 0.6, 0.4 + rb * 0.5)
        accel = max(0, accel)

        # Harsh brake flag probability increases with accel variance and night
        harsh_brake = int(RNG.random() < (0.04 + 0.12 * rb + 0.03 * is_night))

        # Mileage chunk for this sample (proxy for short segment length)
        mileage = max(0.05, RNG.normal(0.8, 0.35)) # ~0.8 miles per record

        rows.append({
            "timestamp": ts.isoformat(),
            "driver_id": driver_id,
            "speed_mph": round(speed, 2),
            "accel_ms2": round(accel, 3),
            "harsh_brake": harsh_brake,
            "mileage": round(mileage, 3),
            "is_night": is_night,
            "road_type": road,
        })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--drivers", type=int, default=500)
    parser.add_argument("--out", type=str, default=os.path.join("data", "trips.csv"))
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    df = synth_trips(args.rows, args.drivers)
    df.to_csv(args.out, index=False)
    # Also write parquet if pyarrow is present (optional)

    try:
        df.to_parquet(args.out.replace(".csv", ".parquet"), index=False)
    except Exception:
        pass

    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()