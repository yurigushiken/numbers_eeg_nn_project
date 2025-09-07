"""Binary classification task for conditions 13 and 31."""
import numpy as np
import pandas as pd

__all__ = ["label_fn"]

VALID_CONDITIONS = {13, 31}

def label_fn(meta: pd.DataFrame):
    cond_int = pd.to_numeric(meta["Condition"], errors='coerce')
    labels = cond_int.where(cond_int.isin(VALID_CONDITIONS), other=np.nan)
    return labels.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
