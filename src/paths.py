"""Centralized output-directory layout for the suite.

All generated artifacts live under output/ in topic subfolders so the results stay organized:

    output/
      benchmark1/        # Mechanistic understanding: CSV, plots, judge report
      benchmark2/        # Ligand ranking: CSV, plots
      ablations/         # Prospective ablation results (A-F) + convergence plots
      constrained/       # Real-world constrained campaigns (batch, divergent, click) + audits
      campaign_details/  # Per-step LLM justification logs for every campaign
      library/           # Enumerated click building blocks & combinatorial library (.smi)
      saturn_mbh/        # MBH de novo campaigns run via external Saturn (config, logs, checkpoints)
      logs/              # Misc run logs

Use out_path(category, filename) to resolve a path (the subfolder is created automatically).
"""

import os

OUTPUT_ROOT = "output"

SUBDIRS = {
    "benchmark1": "benchmark1",
    "benchmark2": "benchmark2",
    "ablations": "ablations",
    "constrained": "constrained",
    "campaign_details": "campaign_details",
    "library": "library",
    "saturn_mbh": "saturn_mbh",
    "logs": "logs",
}


def out_dir(category):
    """Return (creating if needed) the absolute-ish path to an output subfolder."""
    if category not in SUBDIRS:
        raise ValueError(f"Unknown output category: {category}")
    path = os.path.join(OUTPUT_ROOT, SUBDIRS[category])
    os.makedirs(path, exist_ok=True)
    return path


def out_path(category, filename):
    """Resolve output/<category>/<filename>, ensuring the subfolder exists."""
    return os.path.join(out_dir(category), filename)
