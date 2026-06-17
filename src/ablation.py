import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.runner import run_prospective_experiment
from src.paths import out_path

AVAILABLE_ABLATION_MODES = ["A", "B", "C", "D", "E", "F"]

# Every campaign starts from the same known reference catalyst before any optimization step,
# so all trajectories are anchored at (x=0, 37% yield) for a fair common starting point.
START_YIELD = 37.0

STYLE_CONFIGS = {
    "A": {"color": "#4285F4", "label": "A (Baseline / Shuffled SMILES)", "marker": "o"},
    "B": {"color": "#D97757", "label": "B (-SAR Ladder)", "marker": "s"},
    "C": {"color": "#9b59b6", "label": "C (-SMILES Shuffle)", "marker": "d"},
    "D": {"color": "#f1c40f", "label": "D (-Mechanism)", "marker": "<"},
    "E": {"color": "#e67e22", "label": "E (-Chemical Info / Blackout)", "marker": ">"},
    "F": {"color": "#E74C3C", "label": "F (Full Ablation / All Combined)", "marker": "x"},
    # Real-World Constrained Campaigns
    "batch": {"color": "#16a085", "label": "Batch (6 x 7 high-throughput)", "marker": "P"},
    "divergent": {"color": "#8e44ad", "label": "Divergent synthesis", "marker": "*"},
    "click": {"color": "#2c3e50", "label": "Click Library", "marker": "h"},
    "click_agnostic": {"color": "#95a5a6", "label": "Click Library (chem. agnostic)", "marker": "v"},
}


# Shared publication styling so every comparison figure has a consistent, clean look.
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


def _style_publication_axes(ax, ymin, ymax, ystep=10):
    """Apply the shared publication look: white bg, despined frame, light y-grid."""
    ax.set_facecolor("white")
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(range(int(ymin), int(ymax) + 1, ystep))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def run_click_experiment(model="google/gemini-3.5-flash", campaigns=5, steps=14, save_details=False, chem_agnostic=False):
    """Run the SDL click-library campaign: LLM fed building blocks, proposes (alkyne, azide) pairs.

    chem_agnostic=True runs the chemistry-stripped variant (analog of Ablation E) to test whether
    domain knowledge helps library SELECTION the way it was found redundant for de novo DESIGN.
    """
    import concurrent.futures
    from src.runner import run_click_campaign
    os.makedirs("output", exist_ok=True)

    label = "click_agnostic" if chem_agnostic else "click"
    tag = "CHEMISTRY-AGNOSTIC CLICK" if chem_agnostic else "CLICK SDL"
    print(f"\n==================== RUNNING {tag} CAMPAIGN FOR {model} ====================")
    try:
        from src.surrogate import init_surrogate
        surrogate, fp_gen, dataset_dict = init_surrogate()
    except Exception as e:
        print(f"[!] Error initializing surrogate: {e}")
        return None

    runs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=campaigns) as ex:
        futs = {ex.submit(run_click_campaign, i, model, surrogate, fp_gen, dataset_dict,
                          total_iterations=steps, save_details=save_details, chem_agnostic=chem_agnostic): i
                for i in range(1, campaigns + 1)}
        for fut in concurrent.futures.as_completed(futs):
            cid = futs[fut]
            try:
                res = fut.result()
                if res:
                    runs.append(res)
            except Exception as e:
                print(f">> [Failure] {tag} campaign {cid} crashed: {e}")

    with open(out_path("constrained", f"ablation_results_{label}.json"), "w") as f:
        json.dump(runs, f)

    # Single combined plot overlaying all constrained campaigns (no per-campaign plot).
    output_path = plot_constrained_comparison(model_name=model)
    return output_path


def get_click_library_yields(use_cache=True):
    """Return the array of predicted yields for every product in the click library.

    The full-library scoring is moderately expensive (RDKit + surrogate over 324 products), so
    the result is cached to output/constrained/click_library_yields.json after the first run.
    """
    cache_path = out_path("constrained", "click_library_yields.json")
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return np.array(json.load(f), dtype=float)
        except Exception:
            pass
    try:
        from src.click_library import get_library
        from src.surrogate import init_surrogate, score_ligand
        lib = get_library()
        surrogate, fp_gen, dataset_dict = init_surrogate()
        ys = []
        for smi in lib["product_map"].values():
            y, _ = score_ligand(smi, surrogate, fp_gen, dataset_dict, [])
            ys.append(float(y))
        with open(cache_path, "w") as f:
            json.dump(ys, f)
        return np.array(ys, dtype=float)
    except Exception as e:
        print(f"[!] Could not compute click-library yield distribution: {e}")
        return None


def plot_constrained_comparison(model_name="google/gemini-3.5-flash", output_path=None):
    """Overlay the three Real-World Constrained Campaigns (batch, divergent, click) on one figure.

    Each campaign may have a different number of steps (batch is 6x7, divergent/click are 3x14)
    but spends the same 42-reaction budget, so the x-axis is cumulative molecules synthesized
    (proposals/step x step) rather than step index. This makes the 6x7 batch run advance in
    jumps of 6 (2x the 3/step pace) and end at the same x=42 as the 3x14 campaigns. The mean
    +/- SD band is shown for each, against the 89% database-best reference line.
    """
    if output_path is None:
        output_path = out_path("constrained", "generative_active_learning_constrained_comparison.png")

    labels = ["batch", "divergent", "click", "click_agnostic"]
    # Molecules proposed/synthesized per optimization step for each campaign. The batch campaign
    # synthesizes 6 molecules/step (6x7), the others 3/step (3x14); plotting against cumulative
    # molecules synthesized puts every campaign on the same 42-reaction budget axis so the 6x7
    # batch run advances 2x faster (6 vs 3) and ends at the same x as the 3x14 runs.
    PROPOSALS_PER_STEP = {"batch": 6, "divergent": 3, "click": 3, "click_agnostic": 3}
    plt.rcParams.update(PUB_RCPARAMS)
    fig, ax = plt.subplots(figsize=(9, 6.2))

    plotted_any = False
    max_x = 0
    for label in labels:
        results_file = out_path("constrained", f"ablation_results_{label}.json")
        if not os.path.exists(results_file):
            continue
        try:
            with open(results_file) as f:
                runs = json.load(f)
            runs = [r for r in runs if isinstance(r, list)]
            if not runs:
                continue
            arr = np.array(runs)
            if arr.ndim < 2:
                continue
        except Exception:
            continue

        mean_yield = np.mean(arr, axis=0)
        std_yield = np.std(arr, axis=0)
        per_step = PROPOSALS_PER_STEP.get(label, 3)
        molecules = np.arange(1, arr.shape[1] + 1) * per_step

        # Anchor every trajectory at the shared (0 molecules, 37% yield) starting catalyst.
        molecules = np.concatenate(([0], molecules))
        mean_yield = np.concatenate(([START_YIELD], mean_yield))
        std_yield = np.concatenate(([0.0], std_yield))

        success_count = sum(1 for r in runs if max(r) >= 89.0)
        cfg = STYLE_CONFIGS.get(label, {"color": "#888888", "label": label, "marker": "o"})
        lbl = f"{cfg['label']} ({success_count}/{len(runs)} hit 89%)"

        ax.fill_between(molecules, mean_yield - std_yield, mean_yield + std_yield,
                        color=cfg["color"], alpha=0.12, linewidth=0, zorder=2)
        ax.plot(molecules, mean_yield, color=cfg["color"], linestyle="-",
                marker=cfg["marker"], markersize=7, markeredgecolor="white",
                markeredgewidth=0.9, linewidth=2.6, zorder=4, label=lbl)
        max_x = max(max_x, molecules[-1])
        plotted_any = True

    if not plotted_any:
        print("[!] No constrained-campaign results found to plot.")
        return None

    ax.axhline(y=89.0, color="#c0392b", linestyle=":", linewidth=2.0, zorder=3)
    ax.text(max_x, 89.0 + 0.4, "Best database hit (89.0%)", color="#c0392b",
            fontsize=11, fontweight="bold", ha="right", va="bottom")
    ax.set_xlabel("Molecules synthesized", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_ylabel("Maximum discovered yield (%)", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_xlim(-0.5, max_x + 0.5)
    _style_publication_axes(ax, 30, 92, ystep=10)
    ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc", borderpad=0.8)

    # Inset: distribution of all possible yields the click library could produce. Drawn as a
    # horizontal histogram in the right-mid section so its yield axis aligns with the main y-axis,
    # showing that the 89% champion is a rare needle (~0.3% of 324 products), not a biased library.
    click_yields = get_click_library_yields()
    if click_yields is not None and len(click_yields):
        click_color = STYLE_CONFIGS["click"]["color"]
        axin = ax.inset_axes([0.74, 0.30, 0.22, 0.50])  # [x0, y0, w, h] in axes fraction
        bins = np.arange(30, 93, 5)
        axin.hist(click_yields, bins=bins, orientation="horizontal",
                  color=click_color, alpha=0.55, edgecolor="white", linewidth=0.5)
        axin.axhline(89.0, color="#c0392b", linestyle=":", linewidth=1.4)
        axin.axhline(float(np.median(click_yields)), color="#444444",
                     linestyle="--", linewidth=1.2)
        axin.set_ylim(30, 92)
        axin.set_xlabel("Count", fontsize=9, labelpad=2)
        axin.set_ylabel("Library yield (%)", fontsize=9, labelpad=2)
        axin.tick_params(labelsize=8)
        axin.set_yticks(range(30, 91, 20))
        axin.spines["top"].set_visible(False)
        axin.spines["right"].set_visible(False)
        axin.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    print(f"\n-> Constrained-campaign comparison plot saved to '{output_path}' (+ .svg).")
    plt.close()
    return output_path


def run_constrained_experiment(model="google/gemini-3.5-flash", campaigns=5, steps=14, constraint="batch", save_details=False):
    """Run a Real-World Constrained Campaign and save aggregate results + plot.

    constraint='batch'     -> High-throughput batching: 6 proposals/step over 7 steps (same 42-assay budget as A).
    constraint='divergent' -> Divergent synthesis: 3 proposals/step, prompted constraint + RDKit MCS audit.
    """
    os.makedirs("output", exist_ok=True)

    if constraint == "batch":
        label, num_proposals, steps = "batch", 6, 7
        constraint_flag = None
    elif constraint == "divergent":
        label, num_proposals = "divergent", 3
        constraint_flag = "divergent"
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    print(f"\n==================== RUNNING CONSTRAINED CAMPAIGN '{label}' FOR {model} ====================")
    print(f"   Config: {num_proposals} proposals/step x {steps} steps  (constraint flag: {constraint_flag})")

    res_dict = run_prospective_experiment(
        [model], num_campaigns=campaigns, ablation_mode=label,
        total_iterations=steps, save_details=save_details,
        num_proposals=num_proposals, constraint=constraint_flag,
    )
    runs = res_dict[model]

    results_path = out_path("constrained", f"ablation_results_{label}.json")
    with open(results_path, "w") as f:
        json.dump(runs, f)

    # Single combined plot overlaying all constrained campaigns (no per-campaign plot).
    output_path = plot_constrained_comparison(model_name=model)

    # For the divergent constraint, summarize how well the LLM honored the shared-scaffold requirement
    if constraint == "divergent" and save_details:
        try:
            summarize_divergent_audit(campaigns=campaigns)
        except Exception as e:
            print(f"[!] Warning: Failed to summarize divergent audit: {e}")

    return output_path


def summarize_divergent_audit(campaigns=5):
    """Aggregate the per-step divergent-synthesis audit metrics from campaign detail logs.

    Reports three complementary metrics (see compute_scaffold_conservation):
      - MCS conservation     (how much core is shared)
      - Murcko agreement     (fraction sharing the same generic ring scaffold)
      - Mean pairwise Tanimoto (overall chemical similarity)
    """
    metrics = ["mcs_conservation_score", "murcko_scaffold_agreement", "mean_pairwise_tanimoto"]
    agg = {k: [] for k in metrics}
    pass_flags = []           # overall batch PASS flags
    per_campaign = {}

    for c in range(1, campaigns + 1):
        path = out_path("campaign_details", f"campaign_details_divergent_campaign_{c}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            steps = json.load(f)
        c_vals = {k: [] for k in metrics}
        c_pass = []
        for s in steps:
            sc = s.get("scaffold_conservation")
            if not sc:
                continue
            # Only audit steps where the divergent constraint was actually active (best < 80%).
            if sc.get("constraint_active") is False:
                continue
            for k in metrics:
                if sc.get(k) is not None:
                    c_vals[k].append(sc[k])
                    agg[k].append(sc[k])
            if sc.get("divergent_pass") is not None:
                c_pass.append(bool(sc["divergent_pass"]))
                pass_flags.append(bool(sc["divergent_pass"]))
        if any(c_vals.values()):
            entry = {k: (sum(v) / len(v) if v else 0.0) for k, v in c_vals.items()}
            entry["pass_rate"] = (sum(c_pass) / len(c_pass)) if c_pass else 0.0
            per_campaign[c] = entry

    if not any(agg.values()):
        print("[!] No divergent-synthesis audit data found to summarize.")
        return

    overall = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}
    n_steps = len(agg["mcs_conservation_score"])
    overall_pass_rate = (sum(pass_flags) / len(pass_flags)) if pass_flags else 0.0
    # A campaign suite is considered to ENFORCE divergence if >=80% of batches pass.
    enforced = overall_pass_rate >= 0.80

    print("\n-------------------- DIVERGENT SYNTHESIS CONSTRAINT AUDIT --------------------")
    print("   Thresholds: Murcko=1.0 (primary) + (MCS>=0.5 OR Tanimoto>=0.4)")
    print(f"   Steps audited: {n_steps}  across {len(per_campaign)} campaigns")
    for c, vals in sorted(per_campaign.items()):
        print(f"   Campaign {c}: PASS={vals['pass_rate']*100:.0f}%  | "
              f"MCS={vals['mcs_conservation_score']*100:.1f}%  "
              f"Murcko={vals['murcko_scaffold_agreement']*100:.1f}%  "
              f"Tanimoto={vals['mean_pairwise_tanimoto']:.2f}")
    print("   " + "-"*70)
    print(f"   >>> OVERALL batch PASS rate:       {overall_pass_rate*100:.1f}%  ({sum(pass_flags)}/{len(pass_flags)} batches)")
    print(f"   >>> OVERALL MCS conservation:      {overall['mcs_conservation_score']*100:.1f}%  (1.0 = batch is one decorated core)")
    print(f"   >>> OVERALL Murcko agreement:      {overall['murcko_scaffold_agreement']*100:.1f}%  (1.0 = all share same ring scaffold)")
    print(f"   >>> OVERALL mean pairwise Tanimoto:{overall['mean_pairwise_tanimoto']:.2f}    (1.0 = identical fingerprints)")
    print(f"   >>> VERDICT: Divergence {'ENFORCED ✓' if enforced else 'NOT enforced ✗'}  (criterion: >=80% of batches pass)")
    print("------------------------------------------------------------------------------")

    audit_path = out_path("constrained", "divergent_audit_summary.json")
    with open(audit_path, "w") as f:
        json.dump({
            "thresholds": {"murcko_min": 1.0, "mcs_min": 0.5, "tanimoto_min": 0.4, "campaign_pass_rate_min": 0.80},
            "overall_metrics": overall,
            "overall_pass_rate": overall_pass_rate,
            "divergence_enforced": enforced,
            "per_campaign": per_campaign,
            "num_steps": n_steps,
        }, f, indent=2)
    print(f"[+] Divergent audit summary saved to '{audit_path}'.")


def run_all_ablations_experiment(model="google/gemini-3.5-flash", campaigns=5, steps=14, target_mode="all", save_details=False):
    """Run ablation-mode prospective active learning campaigns and save aggregate results."""
    print(f"\n==================== RUNNING ABLATION CAMPAIGNS FOR {model} ====================")
    os.makedirs("output", exist_ok=True)

    ablation_results = {}
    modes = AVAILABLE_ABLATION_MODES.copy()
    if target_mode and target_mode != "all":
        if target_mode in modes:
            modes = [target_mode]
        else:
            raise ValueError(f"Invalid ablation mode: {target_mode}")

    # Dynamically gather any already completed JSON files from output so they are all merged on the plot
    for m in AVAILABLE_ABLATION_MODES:
        json_path = out_path("ablations", f"ablation_results_{m}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # If dictionary, extract the first value that is a list
                        val = next(iter(data.values()))
                        if isinstance(val, list):
                            ablation_results[m] = val
                        else:
                            ablation_results[m] = data
                    else:
                        ablation_results[m] = data
            except Exception:
                pass

    for mode in modes:
        print(f"\n>>> Executing Ablation {mode}...")
        res_dict = run_prospective_experiment([model], num_campaigns=campaigns, ablation_mode=mode, total_iterations=steps, save_details=save_details)
        ablation_results[mode] = res_dict[model]

        temp_file = out_path("ablations", f"ablation_results_{mode}.json")
        with open(temp_file, "w") as f:
            json.dump(ablation_results[mode], f)

    full_results_path = out_path("ablations", "ablation_results_full.json")
    with open(full_results_path, "w") as f:
        json.dump(ablation_results, f)

    # Always produce a single combined plot with every available ablation mode overlaid.
    output_path = out_path("ablations", "generative_active_learning_ablation_comparison.png")
    plot_ablation_comparison(ablation_results, model, steps_count=steps, output_path=output_path)

    return output_path


def plot_ablation_comparison(ablation_results, model_name, steps_count=14, output_path=None,
                             modes_subset=None, title=None):
    """Plot the ablation-mode convergence results and save an output PNG.

    modes_subset: optional list of mode keys (e.g. ["A","B","C","D"]) to restrict the plot to.
    title: optional custom plot title.
    """
    if output_path is None:
        output_path = out_path("ablations", "generative_active_learning_ablation_comparison.png") if len(ablation_results) != 1 else out_path("ablations", f"generative_active_learning_ablation_{next(iter(ablation_results))}.png")

    plt.rcParams.update(PUB_RCPARAMS)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    # Anchor every trajectory at the shared (step 0, 37% yield) starting catalyst.
    steps = np.arange(0, steps_count + 1)

    # Sort available modes logically A to F, optionally restricted to a subset.
    # Any extra keys present (e.g. constrained-campaign labels like 'batch'/'divergent') are appended.
    if modes_subset is not None:
        allowed = modes_subset
    else:
        allowed = list(AVAILABLE_ABLATION_MODES) + [k for k in ablation_results if k not in AVAILABLE_ABLATION_MODES]
    sorted_modes = [m for m in allowed if m in ablation_results]

    for mode in sorted_modes:
        runs = ablation_results[mode]
        if not runs or len(runs) == 0:
            continue
        try:
            arr = np.array(runs)
            if arr.ndim < 2:
                continue
            mean_yield = np.mean(arr, axis=0)
            std_yield = np.std(arr, axis=0)

            # Anchor the trajectory at the shared (step 0, 37% yield) starting catalyst.
            mean_yield = np.concatenate(([START_YIELD], mean_yield))
            std_yield = np.concatenate(([0.0], std_yield))
            
            # Count how many campaigns reached or exceeded 89% yield
            success_count = sum(1 for run in runs if max(run) >= 89.0)
            total_runs = len(runs)
            
            # Calculate the peak yield achieved by each run, and find the worst of these peaks
            peak_yields = [max(run) for run in runs]
            worst_peak = min(peak_yields)
        except Exception:
            continue

        config = STYLE_CONFIGS.get(mode, {"color": "#888888", "label": mode, "marker": "o"})
        label_with_success = f"{config['label']} ({success_count}/{total_runs} hit 89% | worst peak: {worst_peak:.1f}%)"

        # Plot standard deviation range (translucent shading)
        ax.fill_between(steps, mean_yield - std_yield, mean_yield + std_yield,
                        color=config["color"], alpha=0.12, linewidth=0, zorder=2)
        # Plot the main trajectory line
        ax.plot(steps, mean_yield, color=config["color"], linestyle="-",
                marker=config["marker"], markersize=7, markeredgecolor="white",
                markeredgewidth=0.9, linewidth=2.6, zorder=4, label=label_with_success)

    # Ground truth reference line
    ax.axhline(y=89.0, color="#c0392b", linestyle=":", linewidth=2.0, zorder=3)
    ax.text(steps[-1], 89.0 + 0.4, "Best database hit (89.0%)", color="#c0392b",
            fontsize=11, fontweight="bold", ha="right", va="bottom")

    ax.set_xlabel("Optimization step", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_ylabel("Maximum discovered yield (%)", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_xlim(-0.3, steps_count + 0.3)
    ax.set_xticks(steps)

    # Calculate global minimum from all plotted runs (including the 37% start anchor)
    global_min = START_YIELD
    for mode in sorted_modes:
        runs = ablation_results[mode]
        if runs and len(runs) > 0:
            global_min = min(global_min, np.min(runs))
    ymin = 10 * np.floor(min(global_min, START_YIELD) / 10) if global_min < 100.0 else 30
    _style_publication_axes(ax, ymin, 92, ystep=10)

    # Position the legend beautifully inside the bottom right of the plot area
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#cccccc", borderpad=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    print(f"\n-> Ablation plot successfully saved to '{output_path}' (+ .svg).")
    plt.close()
