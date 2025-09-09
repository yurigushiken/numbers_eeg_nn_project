"""Landing-digit decoding for large numbers (4, 5, 6) from large primes.

Predict the ones-place digit (4, 5, or 6) of the experimental **Condition**
code. This task filters the dataset to only include trials where both the prime
and the final stimulus were from the large number group {4, 5, 6}.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn", "CONDITIONS"]

def label_fn(meta: pd.DataFrame):
    """
    Return landing digit (4, 5, or 6) for trials where both prime and stimulus
    are in {4, 5, 6}.
    """
    # Explicitly define all valid conditions where both the prime (tens-place)
    # and the landing digit (ones-place) are in the large number group.
    CONDITIONS = [
        # Landing on 4
        44, 54, 64,
        # Landing on 5
        45, 55, 65,
        # Landing on 6
        46, 56, 66,
    ]
    valid_conditions = CONDITIONS

    cond_int = meta["Condition"].astype(int)

    # Get the landing digit for all trials
    landing_digit = cond_int % 10

    # Keep only the trials where the condition is in our explicit list.
    # Others become NaN and are ignored by the training pipeline.
    valid_landing_digits = landing_digit.where(cond_int.isin(valid_conditions), other=np.nan)

    # Conditionally convert to string, preserving NaNs.
    return valid_landing_digits.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
