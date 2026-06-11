import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.runner import run_prospective_experiment

AVAILABLE_ABLATION_MODES = ["A", "B", "C", "D", "E", "F", "G"]

STYLE_CONFIGS = {
    "A": {"color": "#4285F4", "label": "A (Baseline / Canonical + Full Pipeline)", "marker": "o"},
    "B": {"color": "#D97757", "label": "B (-SAR Ladder & Raw SMILES)", "marker": "s"},
    "C": {"color": "#9b59b6", "label": "C (-Portfolio Directive)", "marker": "d"},
    "D": {"color": "#f1c40f", "label": "D (-Mechanism)", "marker": "<"},
    "E": {"color": "#e67e22", "label": "E (-Chemical Info / Blackout)", "marker": ">"},
    "F": {"color": "#a0522d", "label": "F (-Chem Blackout & -Portfolio)", "marker": "p"},
    "G": {"color": "#E74C3C", "label": "G (Full Ablation / All Combined)", "marker": "x"}
}


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
        json_path = f"output/ablation_results_{m}.json"
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

        temp_file = f"output/ablation_results_{mode}.json"
        with open(temp_file, "w") as f:
            json.dump(ablation_results[mode], f)

    full_results_path = "output/ablation_results_full.json"
    with open(full_results_path, "w") as f:
        json.dump(ablation_results, f)

    if target_mode == "all":
        output_path = "output/generative_active_learning_ablation_comparison.png"
    else:
        output_path = f"output/generative_active_learning_ablation_{target_mode}.png"

    plot_ablation_comparison(ablation_results, model, steps_count=steps, output_path=output_path)
    return output_path


def plot_ablation_comparison(ablation_results, model_name, steps_count=14, output_path=None,
                             modes_subset=None, title=None):
    """Plot the ablation-mode convergence results and save an output PNG.

    modes_subset: optional list of mode keys (e.g. ["A","B","C","D"]) to restrict the plot to.
    title: optional custom plot title.
    """
    if output_path is None:
        output_path = "output/generative_active_learning_ablation_comparison.png" if len(ablation_results) != 1 else f"output/generative_active_learning_ablation_{next(iter(ablation_results))}.png"

    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    steps = np.arange(1, steps_count + 1)

    # Sort available modes logically A to H, optionally restricted to a subset
    allowed = modes_subset if modes_subset is not None else AVAILABLE_ABLATION_MODES
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
        
        # Plot the main trajectory line
        ax.plot(steps, mean_yield, color=config["color"], linestyle="-", 
                marker=config["marker"], markersize=8, linewidth=2.5, label=label_with_success)
        
        # Plot standard deviation range (translucent shading)
        ax.fill_between(steps, mean_yield - std_yield, mean_yield + std_yield, 
                        color=config["color"], alpha=0.10)

    # Ground truth reference line
    ax.axhline(y=89.0, color="#2ca02c", linestyle=":", linewidth=2, label="Best Database Hit (89.0% Limit)")
    
    ax.set_xlabel("Active Learning Optimization Step", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Maximum Discovered Yield (%)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_xticks(steps)
    ax.set_ylim(30, 95)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plot_title = title if title else f"Active Learning Ablation Analysis: {model_name.split('/')[-1]}"
    ax.set_title(plot_title, fontsize=15, fontweight="bold", pad=15)
    
    # Position the legend beautifully inside the bottom right of the plot area
    ax.legend(loc="lower right", fontsize=10.5, frameon=True, facecolor="white", edgecolor="none", shadow=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n-> Ablation plot successfully saved to '{output_path}'.")
    plt.close()


def load_all_ablation_results():
    """Load every available ablation_results_{mode}.json from output/ into a dict."""
    ablation_results = {}
    for m in AVAILABLE_ABLATION_MODES:
        json_path = f"output/ablation_results_{m}.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        val = next(iter(data.values()))
                        ablation_results[m] = val if isinstance(val, list) else data
                    else:
                        ablation_results[m] = data
            except Exception:
                pass
    return ablation_results


def plot_grouped_ablations(model_name="google/gemini-3.5-flash", steps_count=14):
    """Generate the two focused narrative plots from all saved ablation results.

    Plot 1 (A,B,C,G): power of added computational-science (CS) scaffolding tools
                      (SAR ladder, canonical SMILES, portfolio search) -> better
                      reproducibility, lower variance, faster optimization.
    Plot 2 (A,D,E,F): power of chemical rationale, the surprising strength of
                      CS-only search, and that no info beats partial info.
    """
    ablation_results = load_all_ablation_results()

    path1 = "output/ablation_group_CS_tools_ABCG.png"
    plot_ablation_comparison(
        ablation_results, model_name, steps_count=steps_count, output_path=path1,
        modes_subset=["A", "B", "C", "G"],
        title="Impact of Computational Scaffolding Tools (A vs B, C, G)"
    )

    path2 = "output/ablation_group_chemical_knowledge_ADEF.png"
    plot_ablation_comparison(
        ablation_results, model_name, steps_count=steps_count, output_path=path2,
        modes_subset=["A", "D", "E", "F"],
        title="Impact of Chemical Knowledge (A vs D, E, F)"
    )

    return path1, path2
