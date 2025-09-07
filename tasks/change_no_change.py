"""Binary change vs. no-change decoding task.

This task classifies trials into two categories:
- 'change': The stimulus digit is different from the prime digit.
- 'no_change': The stimulus digit is the same as the prime digit (cardinality).
"""
import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """
    Assigns a 'change' or 'no_change' label based on the trial condition.

    Args:
        meta: DataFrame containing trial metadata with a 'Condition' column.

    Returns:
        A Series with the binary labels for each trial. Trials with invalid
        conditions (e.g., condition 99) are mapped to np.nan and ignored.
    """
    cond_int = pd.to_numeric(meta["Condition"], errors='coerce')

    prime = cond_int // 10
    stimulus = cond_int % 10

    # Initialize labels as NaN to ignore invalid trials by default
    labels = pd.Series([np.nan] * len(meta), index=meta.index)

    # A condition is valid only if both prime and stimulus are within the 1-6 range
    valid_mask = (prime.between(1, 6)) & (stimulus.between(1, 6))

    # Assign labels only for the valid conditions
    labels.loc[valid_mask] = np.where(
        prime[valid_mask] == stimulus[valid_mask], 
        'no_change', 
        'change'
    )

    return labels
