"""
2D-CNN training engine (spectrogram input).
This engine uses the unified TrainingRunner.
"""
from __future__ import annotations
from typing import Dict, Callable

from code.training_runner import TrainingRunner
from code.datasets import SpectrogramDataset
from code.model_builders import build_timm_model, build_spectrogram_aug

def run(cfg: Dict, label_fn: Callable):
    """Run training using the unified TrainingRunner."""

    # --- 1. Create the Dataset ---
    dataset = SpectrogramDataset(cfg, label_fn)

    # --- 2. Select the Model Builder ---
    # The model name is read directly from the config by the builder
    model_builder = lambda conf, num_cls: build_timm_model(conf, num_cls)

    # --- 3. Run the Training ---
    runner = TrainingRunner(cfg, label_fn)
    summary = runner.run(
        dataset=dataset,
        groups=dataset.groups,
        class_names=dataset.class_names,
        model_builder=model_builder,
        aug_builder=build_spectrogram_aug,
        input_adapter=None, # Timm models handle (B, 1, H, W) correctly
    )
    
    return summary
