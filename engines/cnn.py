"""
CNN training engine wrapper (time-series EEGNet family).
This engine uses the unified TrainingRunner.
"""
from __future__ import annotations
from typing import Dict, Callable

from code.training_runner import TrainingRunner
from code.datasets import RawEEGDataset
from code.model_builders import RAW_EEG_MODELS, squeeze_input_adapter, build_raw_eeg_aug

def run(cfg: Dict, label_fn: Callable):
    """Run training using the unified TrainingRunner."""
    
    # --- 1. Create the Dataset ---
    dataset = RawEEGDataset(cfg, label_fn)
    
    # --- 2. Select the Model Builder ---
    model_name = cfg.get("model_name", "eegnet")
    if model_name not in RAW_EEG_MODELS:
        raise ValueError(f"Unknown model_name '{model_name}' for the cnn engine. "
                         f"Available models: {list(RAW_EEG_MODELS.keys())}")
    
    # Curry the builder with dataset-specific shapes
    model_builder = lambda conf, num_cls: RAW_EEG_MODELS[model_name](
        conf, num_cls, C=dataset.num_channels, T=dataset.time_points
    )

    # --- 3. Select an Input Adapter (if needed) ---
    # The CwA-Transformer expects (B, C, T) instead of (B, 1, C, T)
    input_adapter = squeeze_input_adapter if model_name == "cwat" else None

    # --- 4. Run the Training ---
    runner = TrainingRunner(cfg, label_fn)
    summary = runner.run(
        dataset=dataset,
        groups=dataset.groups,
        class_names=dataset.class_names,
        model_builder=model_builder,
        aug_builder=lambda conf, d: build_raw_eeg_aug(conf, d.time_points),
        input_adapter=input_adapter,
    )

    return summary
