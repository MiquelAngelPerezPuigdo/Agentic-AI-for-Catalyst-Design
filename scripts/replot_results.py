"""Regenerate Benchmark 1 and Benchmark 2 figures from the EXISTING result CSVs.

This is plot-only: it never calls any model/judge API and never overwrites the
result CSVs. Use it to restyle figures after editing the plotting code in
``src/report.py`` without re-running (and perturbing) the experiments.

Usage:
    python scripts/replot_results.py
"""

import os
import sys

# Make the repo root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report import plot_fair_results, plot_level_hierarchy
from src.paths import out_path

BENCHMARK1_CSV = out_path("benchmark1", "fair_chemistry_results.csv")
BENCHMARK2_CSV = out_path("benchmark2", "benchmark2_results.csv")


def main():
    if os.path.exists(BENCHMARK1_CSV):
        print(f"-> Replotting Benchmark 1 from {BENCHMARK1_CSV}")
        plot_fair_results(BENCHMARK1_CSV)
    else:
        print(f"[!] Skipping Benchmark 1: {BENCHMARK1_CSV} not found.")

    if os.path.exists(BENCHMARK2_CSV):
        print(f"-> Replotting Benchmark 2 from {BENCHMARK2_CSV}")
        plot_level_hierarchy(BENCHMARK2_CSV)
    else:
        print(f"[!] Skipping Benchmark 2: {BENCHMARK2_CSV} not found.")


if __name__ == "__main__":
    main()
