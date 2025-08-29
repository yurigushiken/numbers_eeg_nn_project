"""30-class landing digit task including cardinality.

Predicts the full two-digit condition code (e.g., '11', '12', '21', etc.)
for all 30 valid experimental conditions. This task must be run on an
"all_trials" dataset to have access to the cardinality trials.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

# Defines the 30 valid experimental conditions based on the study design
# (prime +/- 3, within the 1-6 range).
VALID_CONDITIONS = {
    11, 12, 13, 14,
    21, 22, 23, 24, 25,
    31, 32, 33, 34, 35, 36,
    41, 42, 43, 44, 45, 46,
    52, 53, 54, 55, 56,
    63, 64, 65, 66
}

def label_fn(meta: pd.DataFrame):
    """Return the two-digit condition code as the label for the 30 valid classes."""
    cond_int = meta["Condition"].astype(int)

    # Filter to keep only the 30 valid conditions, mapping others to NaN.
    labels = cond_int.where(cond_int.isin(VALID_CONDITIONS), other=np.nan)

    # Convert valid labels to string for classification, keeping NaNs intact
    # so they are filtered out by the training runner.
    return labels.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
