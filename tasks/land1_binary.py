"""Binary label: does the transition land on digit '1'?"""

import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    cond_int = meta["Condition"].astype(int)
    return (cond_int % 10 == 1).astype(int) 