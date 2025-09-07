import argparse
import json
import numpy as np
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt
from typing import List
import mne
from mne.viz import plot_topomap
from mne.viz.topomap import _prepare_topomap_plot
import yaml # <-- Import yaml
from mne.transforms import apply_trans
from mne.viz.topomap import _get_pos_outlines

# Add project root to path to allow for robust module imports
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from code.model_builders import RAW_EEG_MODELS
from code.datasets import RawEEGDataset
from utils.xai import compute_and_plot_attributions
from utils.seeding import seed_everything
from utils.reporting import create_consolidated_reports # Re-use for PDF
import tasks as task_registry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_xai_report(
    run_dir: Path,
    summary_data: dict,
    grand_average_plot_path: Path,
    per_fold_plot_paths: List[Path],
    top_channels: List[str],
    peak_time_window: List[float]
):
    """Generates a consolidated HTML/PDF report for XAI results."""
    report_title = f"XAI Report: {summary_data['hyper']['task']} ({summary_data['hyper']['model_name']})"
    run_id = summary_data['run_id']
    
    # Build a text summary of the findings
    summary_text = (
        f"--- XAI Grand Average Summary ---\n"
        f"Run ID: {run_id}\n"
        f"Task: {summary_data['hyper']['task']}\n"
        f"---------------------------------------\n\n"
        f"Key Findings (Averaged over {len(per_fold_plot_paths)} folds):\n"
        f" - Top 10 Most Important Channels:\n"
        f"     1. {top_channels[0]:<5} 2. {top_channels[1]:<5} 3. {top_channels[2]:<5} 4. {top_channels[3]:<5} 5. {top_channels[4]:<5}\n"
        f"     6. {top_channels[5]:<5} 7. {top_channels[6]:<5} 8. {top_channels[7]:<5} 9. {top_channels[8]:<5} 10. {top_channels[9]:<5}\n\n"
        f" - Peak Importance Time Window: {peak_time_window[0]:.0f}ms - {peak_time_window[1]:.0f}ms\n"
    )

    # Re-use the reporting utility, but with a new HTML structure
    html_output_path = run_dir / "consolidated_xai_report.html"
    
    # Simple image embedding helper
    import base64
    def embed_image(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{report_title}</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            .container {{ max-width: 1200px; margin: auto; }}
            h1, h2 {{ text-align: center; }}
            .summary-text pre {{ background-color: #f4f4f4; padding: 1em; border-radius: 5px; font-size: 1.1em; }}
            .findings-container {{ display: flex; gap: 2em; align-items: center; justify-content: center; margin-bottom: 2em; }}
            .grand-average-plot {{ text-align: center; margin-bottom: 2em; }}
            .grand-average-plot img {{ max-width: 80%; border: 1px solid #ccc; }}
            .plots-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1em; }}
            .plot-pair img {{ max-width: 100%; border: 1px solid #ddd; }}
            .plot-pair p {{ text-align: center; font-weight: bold; margin-top: 0.5em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{report_title}</h1>
            <div class="findings-container">
                <div class="summary-text">
                    <h2>Key Findings</h2>
                    <pre>{summary_text}</pre>
                </div>
                <div class="topoplot">
                    <h2>Channel Importance Topoplot</h2>
                    <img src="{embed_image(run_dir / 'xai_analysis' / 'grand_average_xai_topoplot.png')}" alt="Topoplot">
                </div>
            </div>
            <div class="grand-average-plot">
                <h2>Grand Average Attribution Heatmap</h2>
                <img src="{embed_image(grand_average_plot_path)}" alt="Grand Average Heatmap">
            </div>
            <div class="plots-section">
                <h2>Per-Fold Attribution Heatmaps</h2>
                <div class="plots-grid">
                    {''.join(f'''
                    <div class="plot-pair">
                        <img src="{embed_image(path)}" alt="{path.name}">
                        <p>{path.stem.replace("_xai_heatmap", "").replace("_", " ").title()}</p>
                    </div>
                    ''' for path in per_fold_plot_paths)}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    html_output_path.write_text(html_content, encoding="utf-8")
    print(f"\n -> Consolidated XAI HTML report saved to {html_output_path}")

    # --- Generate PDF ---
    pdf_output_path = html_output_path.with_suffix(".pdf")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            # Use file protocol for local files
            page.goto(f"file://{html_output_path.resolve()}")
            page.pdf(path=str(pdf_output_path), format='A4', print_background=True, margin={'top': '20mm', 'bottom': '20mm'})
            browser.close()
        print(f" -> Consolidated XAI PDF report saved via Playwright to {pdf_output_path}")
    except Exception as e:
        print(f" -> Could not generate PDF report. To enable, run 'pip install playwright && playwright install'. Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run XAI analysis on a completed training run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Path to the run directory.")
    args = parser.parse_args()

    # 1. Load config from summary.json
    try:
        summary_path = next(args.run_dir.glob("summary_*.json"))
        summary_data = json.loads(summary_path.read_text())
    except (StopIteration, FileNotFoundError):
        sys.exit(f"Error: summary_*.json not found in {args.run_dir}")

    cfg = summary_data["hyper"]
    
    # --- NEW: Also load common.yaml for base settings ---
    common_yaml = proj_root / "configs" / "common.yaml"
    if common_yaml.exists():
        common_cfg = yaml.safe_load(common_yaml.read_text()) or {}
        # Merge, with summary's hyper config taking precedence
        common_cfg.update(cfg)
        cfg = common_cfg

    dataset_dir = summary_data["dataset_dir"]
    fold_splits = summary_data["fold_splits"]
    
    # 2. Seed everything for reproducibility of data loading
    seed_everything(cfg.get("seed", 42))

    # 3. Load the full dataset
    label_fn = task_registry.get(cfg["task"])
    # Pass the full config to the dataset so it knows about excluded channels
    cfg_for_dataset = {"dataset_dir": dataset_dir, **cfg}
    full_dataset = RawEEGDataset(cfg_for_dataset, label_fn)
    num_classes = len(full_dataset.class_names)

    # --- NEW (IA Recommended): Get final channel names and sfreq from the dataset ---
    ch_names = full_dataset.channel_names
    sfreq = full_dataset.sfreq
    # --- CORRECTED: Create the actual array of time points ---
    times_ms = np.linspace(0, (full_dataset.time_points - 1) / sfreq * 1000, full_dataset.time_points)

    # --- Build a clean plotting Info object from scratch ---
    montage = mne.channels.read_custom_montage(proj_root / "net/AdultAverageNet128_v1.sfp")
    info_for_plot = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names)
    )
    info_for_plot.set_montage(montage, match_case=False, match_alias=True, on_missing="ignore")

    # 4. Create the nested output directory
    xai_output_dir = args.run_dir / "xai_analysis"
    xai_output_dir.mkdir(exist_ok=True)

    # 5. Find all checkpoints and run XAI
    checkpoints = sorted(list(args.run_dir.glob("fold_*_best.ckpt")))
    if not checkpoints:
        sys.exit("No .ckpt files found in the run directory. Was save_ckpt=true set?")

    print(f"Found {len(checkpoints)} checkpoints. Starting XAI analysis...")

    for fold_info in fold_splits:
        fold_num = fold_info["fold"]
        ckpt_path = args.run_dir / f"fold_{fold_num:02d}_best.ckpt"
        
        if not ckpt_path.exists():
            print(f"--- Skipping Fold {fold_num}: Checkpoint not found ---")
            continue

        print(f"\n--- Processing Fold {fold_num} (Subjects: {fold_info['test_subjects']}) ---")
        
        # Rebuild model and load weights
        model = RAW_EEG_MODELS[cfg["model_name"]](
            cfg, num_classes, C=full_dataset.num_channels, T=full_dataset.time_points
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
        # Find the correct test indices for this fold's subjects
        test_subjects = fold_info["test_subjects"]
        test_indices = [i for i, group in enumerate(full_dataset.groups) if group in test_subjects]
        
        compute_and_plot_attributions(
            model=model,
            dataset=full_dataset,
            test_indices=test_indices,
            device=DEVICE,
            output_dir=xai_output_dir,
            fold_num=fold_num,
            run_dir_name=args.run_dir.name,
            test_subjects=test_subjects,
            ch_names=ch_names,
            times_ms=times_ms,
        )

    print("\n--- XAI Analysis Complete ---")
    
    # --- NEW: Summarization Step ---
    print("\n--- Generating Grand Average Summary and Report ---")

    # 1. Load all attribution .npy files
    all_attributions = []
    npy_files = sorted(xai_output_dir.glob("fold_*_xai_attributions.npy"))
    if not npy_files:
        sys.exit("No attribution .npy files found to summarize.")

    for npy_file in npy_files:
        all_attributions.append(np.load(npy_file))
    
    # 2. Compute Grand Average
    grand_average = np.mean(all_attributions, axis=0)

    # --- NEW (IA Recommended): Add assertion for safety ---
    n_info = len(info_for_plot["ch_names"])
    n_data = grand_average.shape[0]
    if n_info != n_data:
        a = set(info_for_plot["ch_names"])
        b = set(ch_names)
        missing_in_info = sorted(list(b - a))
        missing_in_data = sorted(list(a - b))
        raise RuntimeError(
            f"Topomap mismatch: Info={n_info}, data={n_data}.\n"
            f"Missing in Info: {missing_in_info[:10]} ...\n"
            f"Missing in Data labels: {missing_in_data[:10]} ..."
        )

    # 3. Analyze for Key Findings (metadata is already loaded)
    # Top 10 Channels
    mean_ch_attr = np.mean(np.abs(grand_average), axis=1)
    top_ch_indices = np.argsort(mean_ch_attr)[::-1][:10]
    top_ch_names = [ch_names[i] for i in top_ch_indices]

    # Peak Time Window
    mean_time_attr = np.mean(np.abs(grand_average), axis=0)
    peak_time_idx = np.argmax(mean_time_attr)
    window_size = 50 # ms
    peak_time_ms = times_ms[peak_time_idx]
    time_window = [peak_time_ms - window_size/2, peak_time_ms + window_size/2]

    # 4. Save Grand Average Artifacts
    # The info_for_plot is now correctly created above.

    # Topoplot
    topo_png_path = xai_output_dir / "grand_average_xai_topoplot.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create a mask to highlight the top 10 channels
    mask = np.zeros(len(ch_names), dtype='bool')
    mask[top_ch_indices] = True

    plot_topomap(mean_ch_attr, info_for_plot, axes=ax, show=False, cmap='inferno',
                 mask=mask, mask_params=dict(marker='o', markerfacecolor='w',
                                             markeredgecolor='k', markersize=8))
    ax.set_title('Mean Channel Importance', fontsize=16)

    # --- NEW (Robust): Add labels for top 10 channels ---
    pos, outlines = _get_pos_outlines(info_for_plot, picks=None, sphere=None)

    # Draw a text label for each of the top 10 channels
    for i in top_ch_indices:
        ax.text(pos[i, 0], pos[i, 1], ch_names[i], 
                ha='center', va='center', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    # --- END NEW ---

    fig.savefig(topo_png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    # Heatmap Plot
    ga_png_path = xai_output_dir / "grand_average_xai_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(grand_average, cmap='inferno', aspect='auto', interpolation='nearest')
    ax.set_title('Grand Average Feature Attributions')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('EEG Channels')
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6)
    # Set x-ticks to be in milliseconds
    xtick_indices = np.linspace(0, len(times_ms) - 1, num=10, dtype=int)
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([f"{times_ms[i]:.0f}" for i in xtick_indices])
    fig.colorbar(im, ax=ax, label='Attribution Score')
    plt.tight_layout()
    plt.savefig(ga_png_path)
    plt.close(fig)

    # Npy data
    ga_npy_path = xai_output_dir / "grand_average_xai_attributions.npy"
    np.save(ga_npy_path, grand_average)

    # Summary JSON
    ga_summary_path = xai_output_dir / "grand_average_xai_summary.json"
    summary_data_ga = {
        "run_dir": args.run_dir.name,
        "xai_method": "Integrated Gradients",
        "aggregation": "Grand Average",
        "num_folds_averaged": len(all_attributions),
        "top_10_channels": top_ch_names,
        "peak_time_window_ms": time_window,
        "attribution_data_file": ga_npy_path.name,
        "heatmap_image_file": ga_png_path.name,
        "topoplot_image_file": topo_png_path.name,
    }
    with open(ga_summary_path, 'w') as f:
        json.dump(summary_data_ga, f, indent=2)

    # 5. Generate Consolidated Report
    per_fold_plots = sorted(xai_output_dir.glob("fold_*_xai_heatmap.png"))
    create_xai_report(
        run_dir=args.run_dir,
        summary_data=summary_data,
        grand_average_plot_path=ga_png_path,
        per_fold_plot_paths=per_fold_plots,
        top_channels=top_ch_names,
        peak_time_window=time_window
    )
    
    print("\n--- Grand Average Summary Complete ---")


if __name__ == "__main__":
    main()
