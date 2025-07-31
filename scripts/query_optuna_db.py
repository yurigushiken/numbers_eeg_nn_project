import sqlite3
import json
import sys
from collections import OrderedDict


def get_top_trials(db_path: str, top_n: int = 10):
    """Return list of (trial_id, value, params) for top_n completed trials."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Determine optimisation direction (0=minimize, 1=maximize). Default to maximize if missing
    try:
        direction = cur.execute("SELECT direction FROM studies WHERE study_id=1").fetchone()
        direction = direction[0] if direction is not None else 1
    except sqlite3.Error:
        direction = 1

    order = "DESC" if direction == 1 else "ASC"

    trials = cur.execute(
        f"SELECT trial_id, value FROM trials WHERE state=1 ORDER BY value {order} LIMIT ?",
        (top_n,),
    ).fetchall()

    results = []
    for trial_id, value in trials:
        # Fetch parameters for this trial
        params = OrderedDict(
            cur.execute(
                "SELECT param_name, param_value FROM trial_params WHERE trial_id=? ORDER BY param_name",
                (trial_id,),
            ).fetchall()
        )
        results.append((trial_id, value, params))

    # Extra metadata
    total_completed = cur.execute("SELECT COUNT(1) FROM trials WHERE state=1").fetchone()[0]
    distinct_params = cur.execute("SELECT COUNT(DISTINCT param_name) FROM trial_params").fetchone()[0]

    conn.close()
    return results, total_completed, distinct_params, direction


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_optuna_db.py <path-to-sqlite-db> [top_n]")
        sys.exit(1)

    db_file = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    trials, total_completed, distinct_params, direction = get_top_trials(db_file, top_n)

    print(f"Optimisation direction: {'maximize' if direction == 1 else 'minimize'}")
    print(f"Total completed trials: {total_completed}")
    print(f"Distinct parameters explored: {distinct_params}\n")

    for rank, (tid, val, params) in enumerate(trials, 1):
        print(f"#{rank} Trial {tid} — objective: {val}")
        print(json.dumps(params, indent=2))
        print("-" * 40) 