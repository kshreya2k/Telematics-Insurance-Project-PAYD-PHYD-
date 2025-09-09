# Privacy & Safe Use (POC)


- **Synthetic data only** by default. If uploading data, ensure it is anonymized (no names/addresses/plates/IMEI).
- Keep identifiers **hashed** or **pseudonymized** (e.g., driver_id random UUIDs).
- Collect the minimum viable fields: timestamps, speed, accel, braking flag, mileage, night flag, road type.
- Use **aggregations per driver** for modeling; avoid raw GPS coordinates when not necessary.
- Store files locally in this repo; do not sync to public clouds unless approved.