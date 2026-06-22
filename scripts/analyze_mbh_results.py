"""Summarize Saturn-MBH campaigns: best molecule and score per run.

Auto-discovers every output/saturn_mbh/run_*/oracle_history.csv and prints the
top-scoring molecule for each. Handles both the current
'agentic_mbh_catalyst_score_raw_values' and the legacy
'MBH_catalyst_score_raw_values' columns.

Run:  conda run -n ppchem python scripts/analyze_mbh_results.py
"""
import glob
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, "output", "saturn_mbh")
SCORE_COLS = ("agentic_mbh_catalyst_score_raw_values", "MBH_catalyst_score_raw_values")


def score_column(df: pd.DataFrame) -> str:
    """Return whichever known raw-score column is present."""
    for col in SCORE_COLS:
        if col in df.columns:
            return col
    raise KeyError(f"No known score column in {list(df.columns)}")


def main() -> None:
    rows = []
    for csv in sorted(glob.glob(os.path.join(RUNS_DIR, "run_*", "oracle_history.csv"))):
        df = pd.read_csv(csv)
        col = score_column(df)
        df = df.dropna(subset=[col, "smiles"])
        if df.empty:
            continue
        best = df.sort_values(col, ascending=False).iloc[0]
        rows.append({
            "campaign": os.path.basename(os.path.dirname(csv)),
            "n_scored": len(df),
            "best_score": round(float(best[col]), 1),
            "best_mw": round(float(best.get("mw_raw_values", float("nan"))), 1),
            "best_smiles": best["smiles"],
        })

    out = pd.DataFrame(rows).sort_values("campaign")
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
