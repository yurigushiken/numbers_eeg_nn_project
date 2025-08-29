"""Binary label: is the transition a decrement-by-1?"""

import pandas as pd

__all__ = ["label_fn"]

_DEC_SET = {"65", "54", "43", "32", "21"}

def label_fn(meta: pd.DataFrame):
    vals = meta["Condition"].astype(str)
    return vals.apply(lambda x: 1 if x in _DEC_SET else 0) 