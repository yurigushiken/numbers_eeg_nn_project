#!/usr/bin/env python
"""Export Optuna study plots (PNG) from a saved SQLite database.

Example usage:
    python scripts/export_optuna_plots.py \
        --db optuna_studies/landing_digit_hybrid_nf6-01.db \
        --study landing_digit_hybrid_nf6 \
        --trials 0

If --study is omitted the first study found in the database is used.
The script writes PNGs to results/optuna_plots/<timestamp>_<study_name>/.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Optuna study PNG plots")
    ap.add_argument("--db", required=True, help="SQLite DB file or sqlite:/// URI")
    ap.add_argument("--study", help="Study name (defaults to first one in DB)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/optuna_plots"),
        help="Root output directory (timestamped sub-folder is auto-created)",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Resolve storage URI and study name
    # ------------------------------------------------------------------
    storage_uri = args.db
    if not storage_uri.startswith("sqlite"):
        storage_uri = f"sqlite:///{args.db}"

    if args.study:
        study_name = args.study
    else:
        summaries = optuna.study.get_all_study_summaries(storage=storage_uri)
        if not summaries:
            sys.exit("No studies found in the given Optuna DB.")
        study_name = summaries[0].study_name
        print(f"[info] --study not provided; using first study '{study_name}' in DB.")

    study = optuna.load_study(study_name=study_name, storage=storage_uri)

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = args.out / f"{ts}_{study_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate and save plots
    # ------------------------------------------------------------------
    plot_funcs = {
        "history": plot_optimization_history,
        "parallel": plot_parallel_coordinate,
        "importances": plot_param_importances,
    }

    for name, fn in plot_funcs.items():
        try:
            fig = fn(study)
            # The matplotlib helper returns an Axes for some plots – normalise.
            import matplotlib.pyplot as _plt

            if hasattr(fig, "get_figure"):
                ax = fig  # matplotlib Axes
                fig_obj = ax.figure
            else:
                ax = None
                fig_obj = fig  # already a Figure

            # --------- tweak parallel-coordinate aesthetics ---------
            if name == "parallel":
                # Increase canvas size and adjust label rotation
                fig_obj.set_size_inches(12, 8)
                if ax is None:
                    # For safety, try to grab the first Axes in the Figure
                    ax = fig_obj.axes[0] if fig_obj.axes else None
                if ax is not None:
                    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
                    # Make lines semi-transparent to reduce clutter
                    for line in ax.get_lines():
                        line.set_alpha(0.4)
                    # Lighten grid
                    ax.grid(True, color="#dddddd", linewidth=0.5)

            save_target = fig_obj
            save_target.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
            print(f"[✓] {name}.png")
        except Exception as e:
            print(f"[✗] {name} failed: {e}")

    print(f"All plots saved to {out_dir}")

    # -------------------------------------------------------------
    # Extra variants to improve readability of parallel plots
    # -------------------------------------------------------------
    try:
        from optuna.importance import get_param_importances
        import plotly.io as _pio
        import optuna.visualization as _vis

        # --- choose top-N important params ---
        imp = get_param_importances(study)
        important_params = list(imp.keys())[:6] if imp else None

        # Variant A – only important parameters (Matplotlib)
        if important_params:
            fig = plot_parallel_coordinate(study, params=important_params)
            fig.figure.set_size_inches(12, 8)
            fig.figure.savefig(out_dir / "parallel_top_params.png", dpi=300, bbox_inches="tight")
            print("[✓] parallel_top_params.png")

        # Variant B – best 30 trials & important params
        try:
            top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:30]
            fig = plot_parallel_coordinate(study, params=important_params, include_trials=top_trials)  # type: ignore[arg-type]
            fig.figure.set_size_inches(12, 8)
            fig.figure.savefig(out_dir / "parallel_top_trials.png", dpi=300, bbox_inches="tight")
            print("[✓] parallel_top_trials.png")
        except TypeError:
            # Older Optuna without include_trials
            pass

        # Variant C – interactive Plotly converted to PNG
        try:
            plt_fig = _vis.plot_parallel_coordinate(study, params=important_params)
            _pio.write_image(plt_fig, out_dir / "parallel_plotly.png", scale=2)
            print("[✓] parallel_plotly.png")
        except Exception as e:
            print(f"[✗] parallel_plotly failed: {e}")

        # Variant D – rank plot (Plotly) – less cluttered
        try:
            rk_fig = _vis.plot_rank(study, params=important_params)
            _pio.write_image(rk_fig, out_dir / "rank.png", scale=2)
            print("[✓] rank.png")
        except Exception as e:
            print(f"[✗] rank plot failed: {e}")

    except Exception as e:
        print(f"[!] Extra plot variants skipped due to error: {e}")


if __name__ == "__main__":
    main() 