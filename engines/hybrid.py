"""
Hybrid CwA-Transformer training engine wrapper.

This engine is a special case of the 'cnn' engine and uses the same underlying
TrainingRunner. It is kept for backward compatibility with existing configs
and scripts.
"""
from __future__ import annotations
from typing import Dict, Callable

# Import the refactored cnn engine's run function
from . import cnn

def run(cfg: Dict, label_fn: Callable):
    """
    Run training for the CwA-Transformer model by delegating to the CNN engine.
    """
    # Explicitly set the model_name to 'cwat' to ensure the correct model is used.
    cfg['model_name'] = 'cwat'
    
    # Delegate to the standard CNN engine runner
    return cnn.run(cfg, label_fn)
