"""Clean Ablation-A trajectory plot: the five individual campaigns as dashed lines plus the
bold mean, in the same visual style as the ablation comparison figure. No molecules, stars,
or annotations -- intended as a base figure to be annotated by hand (e.g. in PowerPoint).

Run:
    conda run -n ppchem python scripts/plot_ablationA_campaigns.py
Output:
    output/ablations/ablationA_campaigns_mean.png
"""

import os
import sys
import glob
import json

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from src.paths import out_path

START_YIELD = 37.0


def campaign_trajectory(logs):
    """Per-step running-max yield trajectory, anchored at (step 0, 37%)."""
    cur = 0.0
    traj = []
    for s in logs:
        cur = max(cur, max(p["yield"] for p in s["proposals"]))
        traj.append(cur)
    return np.concatenate(([START_YIELD], traj))


def main():
    files = sorted(glob.glob(out_path("campaign_details", "campaign_details_A_campaign_*.json")))
    if not files:
        raise SystemExit("[!] No A campaign_details found. Run the A campaign with --save-details first.")

    campaigns = {}
    for f in files:
        cid = int(f.split("_campaign_")[-1].split(".")[0])
        campaigns[cid] = json.load(open(f))

    # Common step axis; pad early-stopped runs with their final value.
    max_steps = max(len(logs) for logs in campaigns.values())
    x = np.arange(0, max_steps + 1)
    trajs = {}
    for cid, logs in campaigns.items():
        t = campaign_trajectory(logs)
        if len(t) < len(x):
            t = np.concatenate([t, np.full(len(x) - len(t), t[-1])])
        trajs[cid] = t
    stack = np.vstack(list(trajs.values()))
    mean_traj = np.mean(stack, axis=0)
    std_traj = np.std(stack, axis=0)

    # ---- Publication styling ----
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_facecolor("white")

    # Five individual campaigns as thin dashed grey lines (context, not the message).
    for i, cid in enumerate(sorted(trajs)):
        ax.plot(x, trajs[cid], color="#b8b8b8", linestyle=(0, (5, 3)), linewidth=1.3,
                marker="o", markersize=4, markerfacecolor="white", markeredgewidth=0.9,
                markeredgecolor="#b8b8b8", alpha=0.95, zorder=2,
                label="Individual campaigns ($n=5$)" if i == 0 else None)

    # Bold mean trajectory.
    ax.plot(x, mean_traj, color="#1a4fb0", linestyle="-", linewidth=3.0,
            marker="o", markersize=7, markerfacecolor="#1a4fb0",
            markeredgecolor="white", markeredgewidth=1.0, zorder=5,
            label="Mean trajectory")

    # Database-best reference line.
    ax.axhline(89.0, color="#c0392b", linestyle=":", linewidth=2.0, zorder=4)
    ax.text(max_steps, 89.0 + 0.6, "Best database hit (89.0%)", color="#c0392b",
            fontsize=11, fontweight="bold", ha="right", va="bottom")

    ax.set_xlabel("Optimization step", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_ylabel("Maximum discovered yield (%)", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_xlim(-0.3, max_steps + 0.3)
    ax.set_ylim(START_YIELD - 7, 92)
    ax.set_xticks(range(0, max_steps + 1))
    ax.set_yticks(range(30, 91, 10))

    # Light horizontal-only grid, clean despined frame.
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc", borderpad=0.8)

    plt.tight_layout()
    output_path = out_path("ablations", "ablationA_campaigns_mean.png")
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.close()
    print(f"-> Ablation-A campaigns+mean plot saved to '{output_path}' (+ .svg).")


if __name__ == "__main__":
    main()
