"""
One-time script to preprocess the curated .fif files into a fast-loading
format (.npy and .feather) specifically for the DualStreamCNN engine.

This script reads the output of 01_prepare_for_nn.py and converts it
into a more efficient format for training, dramatically speeding up startup
and data loading times during training.
"""
import argparse
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from tqdm import tqdm

# Add project root to path to allow imports from code
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore", category=UserWarning, module='mne')

def generate_spectrogram(ts_tensor: torch.Tensor, n_fft: int = 128, hop_length: int = 16) -> torch.Tensor:
    """Generates a log-spectrogram from a multi-channel time-series tensor."""
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,
    )
    spec = spectrogram_transform(ts_tensor)
    log_spec = torch.log(spec + 1e-6)
    return log_spec

def main(args):
    """Main processing function."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        return

    print(f"--- Preprocessing for Dual-Stream Engine ---")
    print(f"Input FIF directory: {input_dir}")
    print(f"Output directory:    {output_dir}")

    # Create output directories
    ts_dir = output_dir / "ts_data"
    spec_dir = output_dir / "spec_data"
    ts_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and concatenate all subject .fif files
    print("\nStep 1: Loading and concatenating all .fif files...")
    fif_files = sorted(input_dir.glob("sub-*-epo.fif"))
    if not fif_files:
        print(f"Error: No 'sub-*-epo.fif' files found in {input_dir}")
        return

    epochs_list = []
    import re
    sid_re = re.compile(r"sub-(\d+)")
    for f in fif_files:
        ep = mne.read_epochs(f, preload=True, verbose=False)
        sid_match = sid_re.search(f.name)
        if sid_match:
            sid = int(sid_match.group(1))
            if 'subject' not in ep.metadata.columns:
                ep.metadata['subject'] = sid
            # Ensure subject ID is consistent if it already exists
            ep.metadata['subject'] = ep.metadata['subject'].fillna(sid).astype(int)
        epochs_list.append(ep)

    all_epochs = mne.concatenate_epochs(epochs_list)
    print(f"Found {len(all_epochs)} total trials across {len(fif_files)} subjects.")

    # 2. Extract and save metadata
    print("\nStep 2: Extracting and saving metadata...")
    metadata = all_epochs.metadata.copy()
    metadata.reset_index(drop=True, inplace=True)
    metadata['trial_idx'] = metadata.index
    
    # Ensure all columns are serializable for feather
    for col in metadata.columns:
        if metadata[col].dtype == 'object':
            metadata[col] = metadata[col].astype(str)
            
    metadata_path = output_dir / "metadata.feather"
    metadata.to_feather(metadata_path)
    print(f"Metadata for {len(metadata)} trials saved to {metadata_path}")

    # 3. Extract, process, and save trial data
    print("\nStep 3: Processing and saving individual trials as .npy files...")
    all_data = all_epochs.get_data(copy=False).astype(np.float32) * 1e6 # V to uV
    if all_data.shape[1] > 128:
        all_data = all_data[:, :128, :]

    for idx in tqdm(range(len(all_epochs)), desc="Saving trials"):
        # Get time-series data
        ts_data = all_data[idx] # (C, T)
        
        # Save time-series data
        ts_path = ts_dir / f"{idx}.npy"
        np.save(ts_path, ts_data)
        
        # Generate and save spectrogram
        ts_tensor = torch.from_numpy(ts_data)
        spec_tensor = generate_spectrogram(ts_tensor, n_fft=128, hop_length=16)
        spec_data = spec_tensor.numpy()
        
        spec_path = spec_dir / f"{idx}.npy"
        np.save(spec_path, spec_data)

    print("\n--- Preprocessing complete! ---")
    print(f"Time-series data saved in: {ts_dir}")
    print(f"Spectrogram data saved in:  {spec_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess .fif files for the Dual-Stream engine.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the input .fif files (e.g., data_preprocessed/acc_1_dataset).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the processed .npy and .feather files (e.g., data_dual_stream/acc_1_dataset).")
    
    # For testing in an interactive environment like VSCode, you can uncomment these lines:
    # args = parser.parse_args([
    #     "--input-dir", "data_preprocessed/acc_1_dataset",
    #     "--output-dir", "data_dual_stream/acc_1_dataset"
    # ])
    # main(args)
    
    args = parser.parse_args()
    main(args)
