# src/data_gen/generate.py
import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
ROAD_TYPES = ["highway", "urban", "rural"]

# Center around New Haven, CT (arbitrary for synthetic data)
CITY_LAT, CITY_LON = 41.3083, -72.9279

def is_peak_hour(hour: int) -> int:
    # Morning 7–10, Evening 16–19
    return int((7 <= hour <= 10) or (16 <= hour <= 19))

def jitter_deg(scale_lat=0.05, scale_lon=0.05):
    # small random offsets in degrees (~a few km)
    return RNG.normal(0, scale_lat), RNG.normal(0, scale_lon)

def synth_trips(n_rows: int = 50_000, n_drivers: int = 500) -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    driver_ids = [f"D{str(i).zfill(5)}" for i in range(n_drivers)]

    driver_risk = pd.DataFrame({
        "driver_id": driver_ids,
        "risk_bias": RNG.uniform(0, 1, size=n_drivers),
    })

    rows = []
    for _ in range(n_rows):
        d = driver_risk.sample(1).iloc[0]
        driver_id = d.driver_id
        rb = d.risk_bias

        ts = start + timedelta(minutes=int(RNG.integers(0, 180 * 24 * 60)))
        hour = ts.hour
        is_night = int(hour < 6 or hour >= 22)
        is_peak = is_peak_hour(hour)

        road = RNG.choice(ROAD_TYPES, p=[0.45, 0.4, 0.15])
        base_speed = {
            "highway": RNG.normal(65, 7),
            "urban": RNG.normal(30, 6),
            "rural": RNG.normal(45, 8),
        }[road]
        speed = max(0, base_speed + RNG.normal(0, 5) + rb * 10 * RNG.uniform(-0.2, 1.0))

        accel = RNG.normal(0.8 + rb * 0.6, 0.4 + rb * 0.5)
        accel = max(0, accel)

        harsh_brake = int(RNG.random() < (0.04 + 0.12 * rb + 0.03 * is_night + 0.02 * is_peak))

        mileage = max(0.05, RNG.normal(0.8, 0.35))  # ~0.8 miles per record

        # synthetic lat/lon as a small random walk around city center
        dlat, dlon = jitter_deg()
        lat = CITY_LAT + dlat
        lon = CITY_LON + dlon

        rows.append({
            "timestamp": ts.isoformat(),
            "driver_id": driver_id,
            "speed_mph": round(speed, 2),
            "accel_ms2": round(accel, 3),
            "harsh_brake": harsh_brake,
            "mileage": round(mileage, 3),
            "is_night": int(is_night),
            "is_peak": int(is_peak),
            "road_type": road,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
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
    try:
        df.to_parquet(args.out.replace(".csv", ".parquet"), index=False)
    except Exception:
        pass

    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
