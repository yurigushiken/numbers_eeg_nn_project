"""
Training engine for the Dual-Stream CNN.
"""
from __future__ import annotations
from typing import Dict, Callable

from code.training_runner import TrainingRunner
from code.datasets_dual_stream import DualStreamPreprocessedDataset as DualStreamDataset
from code.model_builders import build_dual_stream_cnn, build_raw_eeg_aug

def run(cfg: Dict, label_fn: Callable):
    """Run training using the unified TrainingRunner."""
    
    # --- 1. Create the Dataset ---
    dataset = DualStreamDataset(cfg, label_fn)
    
    # --- 2. Curry the Model Builder ---
    # The builder will be called by the TrainingRunner with the final number of classes.
    model_builder = lambda conf, num_cls: build_dual_stream_cnn(
        conf, num_cls, C=dataset.num_channels, T=dataset.time_points
    )

    # --- 3. Run the Training ---
    # The TrainingRunner is now flexible enough to handle the dual-stream data.
    runner = TrainingRunner(cfg, label_fn)
    summary = runner.run(
        dataset=dataset,
        groups=dataset.groups,
        class_names=dataset.class_names,
        model_builder=model_builder,
        aug_builder=lambda conf, d: build_raw_eeg_aug(conf, d.time_points),
    )

    return summary
