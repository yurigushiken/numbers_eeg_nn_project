"""Cardinality decoding task.

Predict the digit (1–6) for trials where the prime and stimulus are identical.
These are "no-change" trials, represented by condition codes like 11, 22, ..., 66.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """Return the digit for cardinality conditions (e.g., 11, 22).

    This function identifies trials where the prime number (tens digit) is the
    same as the stimulus number (ones digit) in the 'Condition' code. Only
    trials with codes 11, 22, 33, 44, 55, or 66 are included.
    """
    cond_int = meta["Condition"].astype(int)
    prime = cond_int // 10
    stimulus = cond_int % 10

    # Keep only trials where prime == stimulus and the digit is in the 1-6 range.
    # Other trials will be mapped to NaN and ignored.
    cardinality_trials = stimulus.where((prime == stimulus) & stimulus.between(1, 6), other=np.nan)

    # Convert valid labels to string, but keep NaN values as proper NaN.
    # This ensures they are filtered out by the training runner.
    return cardinality_trials.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
