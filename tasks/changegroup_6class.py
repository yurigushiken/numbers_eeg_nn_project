"""Change-group 6-class decoding task.

Maps the `change_group` column to 6 categories and discards 'NC'.
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    vals = meta["change_group"]
    mask = vals != "NC"
    return vals.where(mask, other=np.nan) 