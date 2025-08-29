"""Decrement-by-1 5-class decoding task."""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    vals = meta["Condition"].astype(str)
    mask = vals.isin(["65", "54", "43", "32", "21"])
    return vals.where(mask, other=np.nan) 