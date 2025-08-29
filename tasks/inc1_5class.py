"""Increment-by-1 5-class decoding task."""

import pandas as pd

__all__ = ["label_fn"]

_INC_CLASSES = ["12", "23", "34", "45", "56"]

def label_fn(meta: pd.DataFrame):
    vals = meta["Condition"].astype(str)
    mask = vals.isin(_INC_CLASSES)
    return vals.where(mask, other=pd.NA) 