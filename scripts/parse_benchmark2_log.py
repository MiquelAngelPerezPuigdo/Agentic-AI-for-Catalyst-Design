"""Reconstruct ``benchmark2_results.csv`` from the raw Benchmark 2 thought-process log.

The Benchmark 2 run saved its full reasoning/scoring transcript to
``output/benchmark2/benchmark2_results.txt`` but the structured results CSV that
``plot_level_hierarchy`` consumes was never written. This script parses that log,
recomputes the per-replicate ``Top_5_Overlap`` exactly as ``src/runner.py`` does
(re-using the same scoring helpers and ground-truth rankings), and writes the CSV
so the figures can be regenerated.

For ``Pd_Dual_Catalysis`` the catalyst set is split across 5 ``CHUNK`` blocks per
iteration; the scores from all chunks of one iteration are merged before scoring,
mirroring a single full ranking.

Usage:
    python scripts/parse_benchmark2_log.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.evaluation import (
    extract_scores_text,
    extract_scores_json,
    calculate_top_k_overlap,
)
from src.ligand_data import RANKING_TASKS
from src.paths import out_path

LOG_PATH = out_path("benchmark2", "benchmark2_results.txt")
CSV_PATH = out_path("benchmark2", "benchmark2_results.csv")

# Overlap depth per task. Pd_Dual_Catalysis has a much larger candidate pool, so
# a top-10 overlap is more informative than the default top-5 used elsewhere.
TASK_TOP_K = {"Pd_Dual_Catalysis": 10}
DEFAULT_TOP_K = 5

# Matches both the simple header and the chunked (Pd_Dual) header variant.
HEADER_RE = re.compile(
    r"^MODEL:\s*(?P<model>[^|]+?)\s*\|\s*"
    r"TASK:\s*(?P<task>[^|]+?)\s*\|\s*"
    r"LEVEL:\s*(?P<level>.+?)\s*\[(?P<mode>JSON|TEXT) MODE\]\s*\|\s*"
    r"(?:ITERATION:\s*(?P<iter>\d+)"
    r"|ITER:\s*(?P<iter_c>\d+)\s*\|\s*CHUNK:\s*(?P<chunk>\d+)\s*/\s*(?P<nchunks>\d+))",
    re.MULTILINE,
)


def parse_log(log_text):
    """Yield (model, task, level, mode, iteration, body) tuples for each block."""
    headers = list(HEADER_RE.finditer(log_text))
    for i, m in enumerate(headers):
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(log_text)
        body = log_text[body_start:body_end]
        iteration = m.group("iter") or m.group("iter_c")
        yield {
            "model": m.group("model").strip(),
            "task": m.group("task").strip(),
            "level": m.group("level").strip(),
            "mode": m.group("mode"),
            "iteration": int(iteration),
            "body": body,
        }


def main():
    if not os.path.exists(LOG_PATH):
        print(f"[!] Log not found: {LOG_PATH}")
        return

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        log_text = f.read()

    # Accumulate scores per (model, task, level, iteration); Pd_Dual chunks merge here.
    grouped = {}
    for block in parse_log(log_text):
        task = block["task"]
        if task not in RANKING_TASKS:
            continue
        if block["mode"] == "JSON":
            scores = extract_scores_json(block["body"])
        else:
            scores = extract_scores_text(block["body"])
        if not scores:
            continue
        key = (block["model"], task, block["level"], block["iteration"])
        grouped.setdefault(key, {}).update(scores)

    rows = []
    for (model, task, level, iteration), scores in grouped.items():
        task_data = RANKING_TASKS[task]
        id_map = task_data["id_map"]
        ligand_score_pairs = [
            (id_map[pid], score) for pid, score in scores.items() if pid in id_map
        ]
        if not ligand_score_pairs:
            continue
        ligand_score_pairs.sort(key=lambda x: x[1], reverse=True)
        predicted_ranking = [pair[0] for pair in ligand_score_pairs]
        k = TASK_TOP_K.get(task, DEFAULT_TOP_K)
        rows.append({
            "Model": model.split("/")[-1],
            "Task": task,
            "Level": level,
            "Iteration": iteration,
            "Top_5_Overlap": calculate_top_k_overlap(
                predicted_ranking, task_data["true_ranking"], k=k
            ),
            "Top_K": k,
            "Valid_Ligands_Ranked": len(predicted_ranking),
        })

    if not rows:
        print("[!] No score blocks parsed from the log.")
        return

    df = pd.DataFrame(rows).sort_values(["Task", "Level", "Model", "Iteration"])
    df.to_csv(CSV_PATH, index=False)
    print(f"[+] Reconstructed {len(df)} rows -> {CSV_PATH}")
    print(df.groupby(["Task", "Model"])["Top_5_Overlap"].agg(["count", "mean", "std"]))


if __name__ == "__main__":
    main()
