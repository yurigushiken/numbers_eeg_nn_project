import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_fold_performance():
    """
    Analyzes the per-fold (subject) performance across all 12 number-pair 
    classification tasks and saves the analysis to a CSV file.
    """
    # Define output directory
    output_dir = Path("results_analysis/outputs")
    output_dir.mkdir(exist_ok=True)

    # Use a broader pattern to be safe
    run_dirs_pattern = "results/runs/*_numbers_pairs_*_cnn_all_trials_dataset (45hz) V2"
    run_paths = list(Path(".").glob(run_dirs_pattern))

    if not run_paths:
        print(f"No run directories found matching the pattern: {run_dirs_pattern}")
        return

    print(f"Found {len(run_paths)} run directories for analysis.")

    fold_accuracies = {} # {fold_num: [acc1, acc2, ...]}

    for run_path in run_paths:
        summary_files = list(run_path.glob("summary_*.json"))
        if not summary_files:
            print(f"Warning: No summary file found in {run_path}")
            continue
        
        summary_path = summary_files[0]
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        per_fold_list = data.get("per_fold_class_metrics")

        if per_fold_list:
            for fold_result in per_fold_list:
                fold_num = fold_result.get("fold")
                fold_acc = fold_result.get("classification_report", {}).get("accuracy")
                
                if fold_num is not None and fold_acc is not None:
                    if fold_num not in fold_accuracies:
                        fold_accuracies[fold_num] = []
                    fold_accuracies[fold_num].append(fold_acc)

    if not fold_accuracies:
        print("No per-fold accuracy data found.")
        return

    fold_data = [
        {
            "Fold": k, 
            "Average Accuracy": np.mean(v)
        } 
        for k, v in fold_accuracies.items() if v
    ]
    
    if not fold_data:
        print("No valid per-fold data to process.")
        return

    df_folds = pd.DataFrame(fold_data)
    
    df_folds_sorted = df_folds.sort_values(by="Average Accuracy", ascending=False).reset_index(drop=True)
    df_folds_sorted.index.name = "Rank"
    df_folds_sorted.index = df_folds_sorted.index + 1
    
    # Save to CSV
    csv_path_folds = output_dir / "analyze_fold_performance.csv"
    df_folds_sorted.to_csv(csv_path_folds, float_format="%.4f")
    print(f"\nSaved per-fold performance analysis to: {csv_path_folds}")
    
    # Display formatted output
    df_folds_sorted["Average Accuracy"] = df_folds_sorted["Average Accuracy"].apply(lambda x: f"{x:.2%}")
    print("\n--- Average Performance by Fold (Subject) ---")
    print(df_folds_sorted.to_string())

if __name__ == "__main__":
    analyze_fold_performance()
