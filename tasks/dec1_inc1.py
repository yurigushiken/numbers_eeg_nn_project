"""Binary label: decrement-1 vs increment-1.

Returns 0 for dec1 (65,54,43,32,21), 1 for inc1 (12,23,34,45,56), and NaN for other transitions so they are ignored.
"""

import pandas as pd

__all__ = ["label_fn"]

_INC_SET = {"12", "23", "34", "45", "56"}
_DEC_SET = {"65", "54", "43", "32", "21"}

def label_fn(meta: pd.DataFrame):
    cond = meta["Condition"].astype(str)

    def _map(code: str):
        if code in _INC_SET:
            return 1
        if code in _DEC_SET:
            return 0
        return pd.NA  # drop unrelated transitions

    return cond.map(_map) 