"""Pairwise decoding task: Landing Digit 1 vs 3."""
import numpy as np
import pandas as pd
__all__ = ["label_fn", "CONDITIONS"]
CONDITIONS = [11, 21, 31, 41, 13, 23, 33, 43, 53, 63]
def label_fn(meta: pd.DataFrame):
    valid_conditions = CONDITIONS
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10
    valid_landing_digits = landing_digit.where(cond_int.isin(valid_conditions), other=np.nan)
    return valid_landing_digits.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
