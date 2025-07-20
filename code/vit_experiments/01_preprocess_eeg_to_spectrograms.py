import os
import sys
from pathlib import Path
import argparse
import json
import mne
import numpy as np
import pandas as pd
from scipy import signal
import pywt  # Continuous Wavelet Transform

"""Stage-1: convert per-trial 128-channel EEG epochs (.fif) into fixed-size
spectrogram tensors and save them under data_spectrograms/…

Usage (run once):
    python 01_preprocess_eeg_to_spectrograms.py \
        --input_dir data_preprocessed/acc_1_dataset \
        --output_root data_spectrograms/landing_digit_cwt_128x128 \
        --img_size 128

Creates one .npy file per trial (shape: 128×H×W, channels-first) and a
metadata.csv mapping each file to labels (landing_digit, Condition, subject).
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default='data_preprocessed/acc_1_dataset',
                   help='Folder with *_preprocessed-epo.fif files')
    p.add_argument('--output_root', default='data_spectrograms/landing_digit_cwt_128x128',
                   help='Where .npy files & metadata.csv will be written')
    p.add_argument('--img_size', type=int, default=128,
                   help='Spectrogram height/width after padding/cropping')
    p.add_argument('--norm', choices=['minmax', 'zscore'], default='minmax',
                   help='Normalisation strategy: per-image minmax (default) or global Z-score')
    p.add_argument('--method', choices=['stft', 'cwt'], default='cwt',
                   help='Time-frequency transform to use (default: cwt)')
    p.add_argument('--fs', type=float, default=256.0, help='Sampling rate (Hz)')
    p.add_argument('--low', type=float, default=1.0, help='Low-cut for STFT output (Hz)')
    p.add_argument('--high', type=float, default=50.0, help='High-cut for STFT output (Hz)')
    return p.parse_args()


def spectrogram_2d(x: np.ndarray, fs: float, img_size: int, f_lo: float, f_hi: float, *, per_image_norm: bool = True):
    """Compute log-power STFT spectrogram of 1-D signal.

    If *per_image_norm* is True the output is scaled to [0,1] by min–max; else
    absolute power values are returned (still log-scaled), merely clipped/ padded
    to img_size.
    """
    # dynamic STFT window so we never violate noverlap < nperseg
    target_nperseg = 256
    nperseg = min(target_nperseg, len(x))
    # 75% overlap but strictly less than nperseg
    noverlap = int(nperseg * 0.75)
    if noverlap >= nperseg:
        noverlap = nperseg - 1 if nperseg > 1 else 0
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    # Limit frequency band
    freq_mask = (f >= f_lo) & (f <= f_hi)
    mag = mag[freq_mask]
    # Log scale power
    spec = 10 * np.log10(mag + 1e-10)
    if per_image_norm:
        spec -= spec.min()
        rng = spec.max() - spec.min()
        if rng != 0:
            spec /= rng
    # Resize to img_size×img_size by padding/cropping
    h, w = spec.shape
    pad_h = max(0, img_size - h)
    pad_w = max(0, img_size - w)
    spec = np.pad(spec, ((0, pad_h), (0, pad_w)), mode='constant')
    spec = spec[:img_size, :img_size]
    return spec.astype(np.float32)


# -----------------------------------------------------------------------------
# Time-frequency transforms
# -----------------------------------------------------------------------------

def cwt_spectrogram_2d(x: np.ndarray, fs: float, img_size: int, f_lo: float, f_hi: float, *, per_image_norm: bool = True):
    """Compute CWT scalogram.

    We use Morlet wavelets with scales chosen to give a roughly logarithmic
    spacing of frequencies between *f_lo* and *f_hi* Hz.  Returned array is
    normalised the same way as the STFT version for consistency.
    """
    # Pick ~img_size frequency bins between f_lo and f_hi on a log scale.
    num_scales = img_size
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), num_scales)
    # Convert desired centre frequencies to wavelet scales.
    # Formula: scale = fs / (freq * wavelet_centre_frequency)
    # Morlet wavelet has centre_frequency ~0.8125 for w=6  (see pywt docs)
    centre_freq = pywt.central_frequency('morl')
    scales = (centre_freq * fs) / freqs

    coefs, _ = pywt.cwt(x, scales=scales, wavelet='morl', sampling_period=1/fs)
    power = np.abs(coefs)

    # Restrict to exact frequency range (already ensured by construction, but
    # keep placeholder for future flexibility)
    # power shape: (num_scales, len(x))

    spec = 10 * np.log10(power + 1e-10)

    if per_image_norm:
        spec -= spec.min()
        rng = spec.max() - spec.min()
        if rng != 0:
            spec /= rng

    # Resize/pad to (img_size, img_size)
    h, w = spec.shape
    pad_h = max(0, img_size - h)
    pad_w = max(0, img_size - w)
    spec = np.pad(spec, ((0, pad_h), (0, pad_w)), mode='constant')
    spec = spec[:img_size, :img_size]
    return spec.astype(np.float32)


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    running_sum = 0.0
    running_sumsq = 0.0
    running_N = 0
    # NOTE: We keep existing .npy files; reruns simply rebuild metadata.csv.
    # This avoids heavy recomputation when only the CSV is missing.

    records = []
    trial_idx = 0
    subj_re = None

    fif_files = sorted(in_dir.glob('sub-*preprocessed-epo.fif'))
    if not fif_files:
        sys.exit(f'No .fif files found in {in_dir}')

    for fif_fp in fif_files:
        print(f'Loading {fif_fp.name}…', flush=True)
        ep = mne.read_epochs(fif_fp, preload=True, verbose=False)
        subject_id = int(fif_fp.name.split("_")[0].split('-')[1])

        data = ep.get_data()            # shape: (n_epochs, n_channels, n_times)
        meta = ep.metadata.copy()       # pandas DF, aligns with epochs
        if data.shape[1] != 128:
            print(f'Warning: {fif_fp.name} has {data.shape[1]} channels (expected 128)')

        for i in range(len(ep)):
            # Pre-compute output filename for this trial
            fname = f'trial_{trial_idx+1:06d}.npy'
            out_fp = out_root / fname

            # Fast path: trial already processed previously
            if out_fp.exists():
                # Build metadata record from existing info
                cond_int = int(meta.iloc[i]['Condition'])
                landing_digit = cond_int % 10
                records.append({
                    'filepath': fname,
                    'landing_digit': landing_digit,
                    'Condition': cond_int,
                    'subject': subject_id
                })

                trial_idx += 1
                if trial_idx % 100 == 0:
                    print(f'  processed {trial_idx} trials (cached)', flush=True)
                continue

            # ---------------------------------------------
            # Full computation path – need to build CWT/STFT
            # ---------------------------------------------

            signal_block = data[i]      # (C, n_times)
            # Harmonise to 128 channels: clip or pad with zeros
            if signal_block.shape[0] > 128:
                signal_block = signal_block[:128]
            elif signal_block.shape[0] < 128:
                pad = np.zeros((128 - signal_block.shape[0], signal_block.shape[1]), dtype=signal_block.dtype)
                signal_block = np.vstack([signal_block, pad])

            specs = []
            for ch_sig in signal_block:
                if args.method == 'stft':
                    spec = spectrogram_2d(ch_sig, args.fs, args.img_size, args.low, args.high, per_image_norm=(args.norm == 'minmax'))
                else:
                    spec = cwt_spectrogram_2d(ch_sig, args.fs, args.img_size, args.low, args.high, per_image_norm=(args.norm == 'minmax'))

                # If global Z-score is requested, postpone normalisation until
                # later.  We still collect running statistics now.
                if args.norm == 'zscore':
                    running_sum += spec.sum()
                    running_sumsq += (spec ** 2).sum()
                    running_N += spec.size
                    specs.append(spec)  # raw spec (0..1 range from power scaling)
                else:  # minmax already applied in helper fns
                    specs.append(spec)
            # Stack → (128, H, W)
            trial_idx += 1
            img_tensor = np.stack(specs, axis=0)
            np.save(out_fp, img_tensor)

            cond_int = int(meta.iloc[i]['Condition'])
            landing_digit = cond_int % 10
            records.append({
                'filepath': fname,
                'landing_digit': landing_digit,
                'Condition': cond_int,
                'subject': subject_id
            })
            if trial_idx % 100 == 0:
                print(f'  processed {trial_idx} trials', flush=True)

    # Compute and save global stats if needed
    if args.norm == 'zscore' and running_N > 0:
        g_mean = running_sum / running_N
        g_std = np.sqrt(max(1e-12, running_sumsq / running_N - g_mean ** 2))
        (out_root / 'stats.json').write_text(json.dumps({'mean': float(g_mean), 'std': float(g_std)}))
        print(f'Global mean={g_mean:.4f}  std={g_std:.4f} written to stats.json')

    # Save metadata with explicit error handling
    try:
        pd.DataFrame(records).to_csv(out_root / 'metadata.csv', index=False)
        print('metadata.csv written OK')
    except Exception as e:
        print(f'!! Failed to write metadata.csv: {e}')

    print(f'Finished. Wrote {trial_idx} trials to {out_root}')


if __name__ == '__main__':
    main() 