"""Binary label: is the transition an increment-by-1?"""

import pandas as pd

__all__ = ["label_fn"]

_INC_SET = {"12", "23", "34", "45", "56"}

def label_fn(meta: pd.DataFrame):
    vals = meta["Condition"].astype(str)
    return vals.apply(lambda x: 1 if x in _INC_SET else 0) 