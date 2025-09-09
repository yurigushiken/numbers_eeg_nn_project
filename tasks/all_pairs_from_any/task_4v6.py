"""Pairwise decoding task: Landing Digit 4 vs 6."""
import numpy as np
import pandas as pd
__all__ = ["label_fn", "CONDITIONS"]
CONDITIONS = [14, 24, 34, 44, 54, 64, 36, 46, 56, 66]
def label_fn(meta: pd.DataFrame):
    valid_conditions = CONDITIONS
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10
    valid_landing_digits = landing_digit.where(cond_int.isin(valid_conditions), other=np.nan)
    return valid_landing_digits.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
