"""Binary label: does the transition land on digit '1' (True/1) or not (False/0)?

This is an explicit version of the `land1_binary` task, where all conditions
are listed for clarity.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """
    Returns a binary label: 1 if the landing digit is '1', and 0 otherwise.
    """
    # Define the conditions that correspond to a landing digit of '1'
    conditions_for_1 = [11, 21, 31, 41, 51, 61]

    # Define the conditions for all other landing digits {2, 3, 4, 5, 6}
    conditions_for_0 = [
        # Landing on 2
        12, 22, 32, 42, 52, 62,
        # Landing on 3
        13, 23, 33, 43, 53, 63,
        # Landing on 4
        14, 24, 34, 44, 54, 64,
        # Landing on 5
        15, 25, 35, 45, 55, 65,
        # Landing on 6
        16, 26, 36, 46, 56, 66,
    ]

    cond_int = meta["Condition"].astype(int)
    
    # Create a Series to hold the labels, default to NaN
    labels = pd.Series(np.nan, index=meta.index)

    # Assign 1 for conditions landing on '1'
    labels[cond_int.isin(conditions_for_1)] = 1

    # Assign 0 for all other valid conditions
    labels[cond_int.isin(conditions_for_0)] = 0

    return labels
