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
from .parity_transition_2class import label_fn as parity_transition_2class # noqa: F401
from .cardinality import label_fn as cardinality # noqa: F401
from .landing_digit_30_class import label_fn as landing_digit_30_class # noqa: F401
from .change_no_change import label_fn as change_no_change # noqa: F401
from .numbers_pairs_12_21 import label_fn as numbers_pairs_12_21 # noqa: F401
from .numbers_pairs_13_31 import label_fn as numbers_pairs_13_31 # noqa: F401
from .numbers_pairs_14_41 import label_fn as numbers_pairs_14_41 # noqa: F401
from .numbers_pairs_23_32 import label_fn as numbers_pairs_23_32 # noqa: F401
from .numbers_pairs_24_42 import label_fn as numbers_pairs_24_42 # noqa: F401
from .numbers_pairs_25_52 import label_fn as numbers_pairs_25_52 # noqa: F401
from .numbers_pairs_34_43 import label_fn as numbers_pairs_34_43 # noqa: F401
from .numbers_pairs_35_53 import label_fn as numbers_pairs_35_53 # noqa: F401
from .numbers_pairs_36_63 import label_fn as numbers_pairs_36_63 # noqa: F401
from .numbers_pairs_45_54 import label_fn as numbers_pairs_45_54 # noqa: F401
from .numbers_pairs_46_64 import label_fn as numbers_pairs_46_64 # noqa: F401
from .numbers_pairs_56_65 import label_fn as numbers_pairs_56_65 # noqa: F401
from .no_cross_landing_digit import label_fn as no_cross_landing_digit # noqa: F401
from .land1_binary_explicit import label_fn as land1_binary_explicit # noqa: F401
from .landing_digit_1_3 import label_fn as landing_digit_1_3 # noqa: F401
from .landing_digit_4_6 import label_fn as landing_digit_4_6 # noqa: F401

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
    "parity_transition_2class": parity_transition_2class,
    "cardinality": cardinality,
    "landing_digit_30_class": landing_digit_30_class,
    "change_no_change": change_no_change,
    "numbers_pairs_12_21": numbers_pairs_12_21,
    "numbers_pairs_13_31": numbers_pairs_13_31,
    "numbers_pairs_14_41": numbers_pairs_14_41,
    "numbers_pairs_23_32": numbers_pairs_23_32,
    "numbers_pairs_24_42": numbers_pairs_24_42,
    "numbers_pairs_25_52": numbers_pairs_25_52,
    "numbers_pairs_34_43": numbers_pairs_34_43,
    "numbers_pairs_35_53": numbers_pairs_35_53,
    "numbers_pairs_36_63": numbers_pairs_36_63,
    "numbers_pairs_45_54": numbers_pairs_45_54,
    "numbers_pairs_46_64": numbers_pairs_46_64,
    "numbers_pairs_56_65": numbers_pairs_56_65,
    "no_cross_landing_digit": no_cross_landing_digit,
    "land1_binary_explicit": land1_binary_explicit,
    "landing_digit_1_3": landing_digit_1_3,
    "landing_digit_4_6": landing_digit_4_6,
}


def get(task_name: str):
    """Return the label_fn for a given task name (KeyError if missing)."""
    return TASKS[task_name] 