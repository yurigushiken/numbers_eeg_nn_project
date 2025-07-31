"""Training engine dispatch."""

from .cnn import run as cnn_run  # noqa: F401
from .hybrid import run as hybrid_run  # noqa: F401
from .dual_stream import run as dual_stream_run # noqa: F401
# Try to import the spectrogram engine; skip cleanly if optional deps missing
try:
    from .cnn_spectrogram import run as cnn_spectrogram_run  # noqa: F401
except ModuleNotFoundError:
    cnn_spectrogram_run = None

ENGINES = {
    "cnn": cnn_run,
    "hybrid": hybrid_run,
    "dual_stream": dual_stream_run,
}
if cnn_spectrogram_run is not None:
    ENGINES["cnn_spectrogram"] = cnn_spectrogram_run

def get(engine_name: str):
    return ENGINES[engine_name] 