"""Plot the MBH catalyst score vs oracle calls for an MBH catalyst-design campaign.

Self-contained: run it inside a Saturn run directory (it auto-detects the oracle
history CSV and the *_raw_values score column), producing publication-styled PNG +
SVG figures next to the CSV. Works both for the bundled latest run
(oracle_history_MBH_17.csv, MBH_catalyst_score_*) and for fresh launcher runs
(oracle_history.csv, agentic_mbh_catalyst_score_*).
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 25  # moving-average window (number of evaluations)

PUB_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "savefig.facecolor": "white",
}

SCORE_COLOR = "#1f77b4"
BEST_COLOR = "#c0392b"
REWARD_COLOR = "#2ca02c"


def _find_csv() -> str:
    """Locate the oracle-history CSV in this directory."""
    for name in ("oracle_history_MBH_17.csv", "oracle_history.csv"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(HERE, "oracle_history*.csv")))
    if not hits:
        raise FileNotFoundError(f"No oracle_history*.csv found in {HERE}")
    return hits[0]


def _find_score_column(df: pd.DataFrame) -> str:
    """Find the catalyst-score raw-values column regardless of component name."""
    candidates = [c for c in df.columns if c.endswith("_raw_values") and c != "mw_raw_values"]
    if not candidates:
        raise KeyError(f"No catalyst-score *_raw_values column in {list(df.columns)}")
    # Prefer an MBH-named column if present, else the first non-mw raw-values column.
    for c in candidates:
        if "mbh" in c.lower():
            return c
    return candidates[0]


CSV_PATH = _find_csv()
df = pd.read_csv(CSV_PATH)
df = df.sort_values("oracle_calls").reset_index(drop=True)
df["eval_index"] = np.arange(1, len(df) + 1)

score_col = _find_score_column(df)
score = df[score_col]
reward = df["reward"]

cumulative_max = score.cummax()
score_ma = score.rolling(window=WINDOW, min_periods=1).mean()
reward_ma = reward.rolling(window=WINDOW, min_periods=1).mean()

plt.rcParams.update(PUB_RCPARAMS)
fig, ax1 = plt.subplots(figsize=(9, 6.2))
ax1.set_facecolor("white")

ax1.scatter(
    df["eval_index"], score,
    s=14, alpha=0.25, color=SCORE_COLOR, linewidth=0, zorder=2,
    label="Individual oracle evaluations",
)
ax1.plot(
    df["eval_index"], score_ma,
    color=SCORE_COLOR, lw=2.4, zorder=4,
    label=f"Moving average (n = {WINDOW})",
)
ax1.plot(
    df["eval_index"], cumulative_max,
    color=BEST_COLOR, lw=2.8, zorder=5, drawstyle="steps-post",
    label="Cumulative maximum",
)
ax1.axhline(y=50.0, color="#555555", linestyle=":", linewidth=1.8, zorder=3,
            label="DABCO reference")

over = df[score > 50.0]
ax1.scatter(
    over["eval_index"], over[score_col],
    s=110, facecolors="none", edgecolors="#e08214", linewidths=2.0, zorder=6,
    marker="o",
)

ax1.set_xlabel("Oracle call (cumulative)", fontsize=14, fontweight="bold", labelpad=8)
ax1.set_ylabel("MBH catalyst score", fontsize=14, fontweight="bold",
               color=SCORE_COLOR, labelpad=8)
ax1.tick_params(axis="y", labelcolor=SCORE_COLOR)
ax1.set_xlim(0, len(df) + 1)
ax1.set_ylim(-2, 100)
ax1.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#e6e6e6", zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)

ax2 = ax1.twinx()
ax2.plot(
    df["eval_index"], reward_ma,
    color=REWARD_COLOR, lw=2.0, ls="--", alpha=0.9, zorder=3,
    label=f"Aggregated reward (moving average, n = {WINDOW})",
)
ax2.set_ylabel("Aggregated reward", fontsize=14, fontweight="bold",
               color=REWARD_COLOR, labelpad=8)
ax2.tick_params(axis="y", labelcolor=REWARD_COLOR)
ax2.set_ylim(0, 1)
ax2.spines["top"].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2, labels1 + labels2,
    loc="upper left", fontsize=11, frameon=True, framealpha=0.95,
    facecolor="white", edgecolor="#cccccc", borderpad=0.8,
)

plt.tight_layout()

out_png = os.path.join(HERE, "score_vs_calls.png")
out_svg = os.path.join(HERE, "score_vs_calls.svg")
fig.savefig(out_png, dpi=400, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close()

print(f"Saved {out_png}")
print(f"Saved {out_svg}")
print(f"Total evaluations: {len(df)}")
print(f"Best MBH catalyst score: {score.max():.2f}  (column: {score_col})")
