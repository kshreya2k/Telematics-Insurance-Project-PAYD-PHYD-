from typing import Tuple
import numpy as np
import pandas as pd

def compute_premium(
    risk_score: pd.Series,
    base: float = 800.0,
    coverage_multiplier: float = 1.0, # slider: 0.5–2.0
    deductible_multiplier: float = 1.0, # slider: 0.7–1.3 (higher deductible => lower multiplier)
    safe_discount_threshold: float = 0.20,
    safe_discount_pct: float = 0.10,
) -> pd.Series:
    """Premium = base * (1 + risk) * coverage_mult * deductible_mult with safe-driver discount.
    """
    prem = base * (1.0 + risk_score.clip(0, 1)) * coverage_multiplier * deductible_multiplier
    discount = (risk_score < safe_discount_threshold).astype(float) * safe_discount_pct
    return prem * (1.0 - discount)

def breakdown_row(risk: float, base: float, cov: float, ded: float, thr: float, disc: float) -> Tuple[float, float, float, float, float]:
    raw = base * (1 + risk)
    after_cov_ded = raw * cov * ded
    discount_amt = after_cov_ded * (disc if risk < thr else 0.0)
    final = after_cov_ded - discount_amt
    return raw, after_cov_ded, discount_amt, final, risk