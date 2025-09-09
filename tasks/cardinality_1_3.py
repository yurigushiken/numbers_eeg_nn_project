"""Cardinality decoding for small numbers (1, 2, 3).

Predict the ones-place digit (1, 2, or 3) of the experimental **Condition**
code, restricted to cardinality (no-change) trials within the small set.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn", "CONDITIONS"]

# CONDITIONS exposed for reporting (used as subtitle in consolidated reports)
CONDITIONS = [11, 22, 33]


def label_fn(meta: pd.DataFrame):
    """
    Return landing digit (1, 2, or 3) for cardinality (no-change) trials
    where Condition is one of {11, 22, 33}.
    """
    valid_conditions = CONDITIONS

    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10

    # Keep only trials whose Condition is explicitly listed; others -> NaN
    valid_landing_digits = landing_digit.where(cond_int.isin(valid_conditions), other=np.nan)

    # Convert to string labels while preserving NaNs
    return valid_landing_digits.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


