"""Landing-digit decoding for large numbers (4, 5, 6).

Predict the ones-place digit (4, 5, or 6) of the experimental **Condition**
code. This task filters the dataset to only include trials where the final
stimulus was one of the large numbers.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """
    Return landing digit (4, 5, or 6) from the Condition code, explicitly
    including only trials where the landing digit is 4, 5, or 6.
    """
    # Explicitly define all valid conditions where the landing digit
    # (ones-place) is 4, 5, or 6.
    valid_conditions = [
        # Landing on 4
        14, 24, 34, 44, 54, 64,
        # Landing on 5
        25, 35, 45, 55, 65,
        # Landing on 6
        36, 46, 56, 66,
    ]

    cond_int = meta["Condition"].astype(int)

    # Get the landing digit for all trials
    landing_digit = cond_int % 10

    # Keep only the trials where the condition is in our explicit list.
    # Others become NaN and are ignored by the training pipeline.
    valid_landing_digits = landing_digit.where(cond_int.isin(valid_conditions), other=np.nan)

    # Conditionally convert to string, preserving NaNs.
    return valid_landing_digits.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
