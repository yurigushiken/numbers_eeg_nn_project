"""No-crossover landing-digit decoding task.

Predict the ones-place digit (1–6) of the experimental **Condition** code,
but only for trials where the prime (tens-place) and stimulus (ones-place)
digits are both in the small-number group {1, 2, 3} or both in the
large-number group {4, 5, 6}.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """
    Return landing digit (1–6) from the Condition code, explicitly including only
    non-crossover trials.
    """
    # Explicitly define all valid "no-crossover" conditions.
    # This includes trials within the small-number group {1,2,3} and
    # trials within the large-number group {4,5,6}, including cardinality.
    valid_conditions = [
        # Small group (prime and landing in {1,2,3})
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        # Large group (prime and landing in {4,5,6})
        44, 45, 46,
        54, 55, 56,
        64, 65, 66
    ]

    cond_int = meta["Condition"].astype(int)

    # Get the landing digit for all trials
    landing_digit = cond_int % 10

    # Keep only the trials where the condition is in our explicit list.
    # Others become NaN and are ignored by the training pipeline.
    valid_landing_digits = landing_digit.where(cond_int.isin(valid_conditions), other=np.nan)

    return valid_landing_digits.astype(str)