"""Landing-digit decoding for small numbers (1, 2, 3) from small primes.

Predict the ones-place digit (1, 2, or 3) of the experimental **Condition**
code. This task filters the dataset to only include trials where both the prime
and the final stimulus were from the small number group {1, 2, 3}.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn", "CONDITIONS"]

def label_fn(meta: pd.DataFrame):
    """
    Return landing digit (1, 2, or 3) for trials where both prime and stimulus
    are in {1, 2, 3}.
    """
    # Explicitly define all valid conditions where both the prime (tens-place)
    # and the landing digit (ones-place) are in the small number group.
    CONDITIONS = [
        # Landing on 1
        11, 21, 31,
        # Landing on 2
        12, 22, 32,
        # Landing on 3
        13, 23, 33,
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
