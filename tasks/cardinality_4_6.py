"""Cardinality decoding for large numbers (4, 5, 6).

Predict the ones-place digit (4, 5, or 6) of the experimental **Condition**
code, restricted to cardinality (no-change) trials within the large set.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn", "CONDITIONS"]

# CONDITIONS exposed for reporting (used as subtitle in consolidated reports)
CONDITIONS = [44, 55, 66]


def label_fn(meta: pd.DataFrame):
    """
    Return landing digit (4, 5, or 6) for cardinality (no-change) trials
    where Condition is one of {44, 55, 66}.
    """
    valid_conditions = CONDITIONS

    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10

    # Keep only trials whose Condition is explicitly listed; others -> NaN
    valid_landing_digits = landing_digit.where(cond_int.isin(valid_conditions), other=np.nan)

    # Convert to string labels while preserving NaNs
    return valid_landing_digits.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


