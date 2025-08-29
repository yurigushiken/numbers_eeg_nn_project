"""Binary label: is the condition a 'numbers pair' (1x or x1)?"""

import pandas as pd

__all__ = ["label_fn"]

import re

_PATTERN = re.compile(r"(1\d|\d1)")

def label_fn(meta: pd.DataFrame):
    cond = meta["Condition"].astype(str)
    is_pair = cond.str.match(_PATTERN)
    return is_pair.map({True: 0, False: 1}) 