"""Landing-digit decoding task.

Predict the ones-place digit (1–6) of the experimental **Condition** code.
Only trials whose code ends in 1-6 are retained; others are ignored.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """Return landing digit (1–6) from the Condition code.

    A few legacy exports stored a pre-computed `landing_digit` field that
    sometimes became corrupted (e.g., constant across all trials).  We
    instead recompute the digit from the integer **Condition** code and
    drop any trial whose ones-place is outside 1 … 6.
    """

    # Compute the ones-place digit and keep only the six valid classes.
    # Anything else is mapped to NaN and will be excluded downstream.
    cond_int = meta["Condition"].astype(int)
    digit = cond_int % 10

    # Keep only digits 1–6; anything else becomes NaN (ignored downstream)
    digit = digit.where(digit.between(1, 6), other=np.nan)

    # IMPORTANT: convert valid digits to strings but preserve NaNs as real NaN
    # (avoid creating a literal "nan" class)
    return digit.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)