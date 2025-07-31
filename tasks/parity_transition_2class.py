"""Parity-transition decoding task (2-class).

Classifies trials based on the parity of the prime and stimulus digits.
The two classes are:
- 'even_to_even'
- 'odd_to_odd'

Crossover trials (odd-to-even, even-to-odd) are excluded.

Constraints applied:
- Prime and stimulus digits are between 1 and 6.
- The absolute difference between prime and stimulus is at most 3.
- Cardinality trials (prime == stimulus) are excluded.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """Computes the parity transition label from the 'Condition' code."""

    cond_int = meta["Condition"].astype(int)
    prime = cond_int // 10
    stimulus = cond_int % 10

    # --- Apply experimental constraints ---
    # 1. Digits are within [1, 6]
    valid_digits = prime.between(1, 6) & stimulus.between(1, 6)
    # 2. No cardinality (prime != stimulus)
    no_cardinality = (prime != stimulus)
    # 3. Difference is at most 3
    valid_diff = (prime - stimulus).abs() <= 3

    # Combine all masks
    valid_trials = valid_digits & no_cardinality & valid_diff

    # Determine parity
    is_prime_even = (prime % 2) == 0
    is_stim_even = (stimulus % 2) == 0

    # --- Assign labels based on parity transitions ---
    # Initialize labels as NaN (to be ignored by the pipeline)
    labels = pd.Series(np.nan, index=meta.index, dtype=object)

    # Assign labels only for valid trials that are not crossovers
    labels.loc[valid_trials & is_prime_even & is_stim_even] = "even_to_even"
    labels.loc[valid_trials & ~is_prime_even & ~is_stim_even] = "odd_to_odd"

    # Define desired order for the classes
    class_order = ['odd_to_odd', 'even_to_even']
    
    # Convert to a categorical series with the specified order
    # This ensures that downstream encoders will respect our custom ordering.
    labels_cat = pd.Categorical(labels, categories=class_order, ordered=True)

    return pd.Series(labels_cat, index=meta.index)
