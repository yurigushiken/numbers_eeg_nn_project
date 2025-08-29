"""Direction binary decoding task.

Classify trials as Increasing ("I") or Decreasing ("D").  Trials labelled
"NC" are discarded (label=NaN).
"""

import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    """Return 'I' or 'D' labels; any other trial becomes NaN.

    The raw metadata can spell the column in various ways
    (e.g. 'Direction', 'direction', 'dir').  We accept any
    column name that case-insensitively matches *direction*.
    Values are treated case-insensitively as well.
    """

    # --- locate the direction column (case-insensitive) ---
    col = None
    for c in meta.columns:
        if c.lower() == "direction" or c.lower().startswith("dir"):
            col = c
            break

    if col is None:
        raise KeyError("No 'direction' column found in metadata. Available columns: "
                       + ", ".join(meta.columns))

    vals = meta[col].astype(str).str.upper()
    mask = vals.isin(["I", "D"])
    return vals.where(mask, other=np.nan) 