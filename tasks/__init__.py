"""Registry of decoding tasks.

A *task* corresponds to a specific cognitive decoding label mapping.
Each module defines a `label_fn(meta)` that converts the `epochs.metadata`
DataFrame into target labels (Pandas Series / NumPy array).
"""

from importlib import import_module

# Explicit import so that IDEs & type-checkers see the symbols.
from .landing_digit import label_fn as landing_digit  # noqa: F401
from .direction_binary import label_fn as direction_binary  # noqa: F401
from .changegroup_6class import label_fn as changegroup_6class  # noqa: F401
from .dec1_5class import label_fn as dec1_5class  # noqa: F401
from .dec1_binary import label_fn as dec1_binary  # noqa: F401
from .inc1_5class import label_fn as inc1_5class  # noqa: F401
from .inc1_binary import label_fn as inc1_binary  # noqa: F401
from .land1_binary import label_fn as land1_binary  # noqa: F401
from .numbers_pairs import label_fn as numbers_pairs  # noqa: F401
from .dec1_inc1 import label_fn as dec1_inc1  # noqa: F401
from .parity_transition import label_fn as parity_transition  # noqa: F401
from .parity_transition_3class import label_fn as parity_transition_3class  # noqa: F401

TASKS = {
    "landing_digit": landing_digit,
    "direction_binary": direction_binary,
    "changegroup_6class": changegroup_6class,
    "dec1_5class": dec1_5class,
    "dec1_binary": dec1_binary,
    "inc1_5class": inc1_5class,
    "inc1_binary": inc1_binary,
    "land1_binary": land1_binary,
    "numbers_pairs": numbers_pairs,
    "dec1_inc1": dec1_inc1,
    "parity_transition": parity_transition,
    "parity_transition_3class": parity_transition_3class,
}


def get(task_name: str):
    """Return the label_fn for a given task name (KeyError if missing)."""
    return TASKS[task_name] 