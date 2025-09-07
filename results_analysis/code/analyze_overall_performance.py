import json
import pandas as pd
from pathlib import Path

def analyze_overall_performance():
    """
    Analyzes the overall performance of the 12 number-pair classification tasks
    and saves the analysis to a CSV file.
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

    overall_results = []

    for run_path in run_paths:
        summary_files = list(run_path.glob("summary_*.json"))
        if not summary_files:
            print(f"Warning: No summary file found in {run_path}")
            continue
        
        summary_path = summary_files[0]
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        task_name = data.get("hyper", {}).get("task")
        mean_acc = data.get("mean_acc")
        
        if task_name and mean_acc is not None:
            overall_results.append({"task": task_name, "mean_accuracy": mean_acc / 100.0})

    if not overall_results:
        print("No overall accuracy data found.")
        return

    df_overall = pd.DataFrame(overall_results)
    df_overall_sorted = df_overall.sort_values(by="mean_accuracy", ascending=False)
    
    # Save to CSV
    csv_path_overall = output_dir / "analyze_overall_performance.csv"
    df_overall_sorted.to_csv(csv_path_overall, index=False, float_format="%.4f")
    print(f"\nSaved overall performance analysis to: {csv_path_overall}")
    
    # Display formatted output
    df_overall_sorted["mean_accuracy"] = df_overall_sorted["mean_accuracy"].apply(lambda x: f"{x:.2%}")
    print("\n--- Overall Performance by Number Pair ---")
    print(df_overall_sorted.to_string(index=False))

if __name__ == "__main__":
    analyze_overall_performance()
