import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.runner import run_prospective_experiment

AVAILABLE_ABLATION_MODES = ["A", "B", "C", "D", "E", "F", "G", "H"]

STYLE_CONFIGS = {
    "A": {"color": "#4285F4", "label": "A (Baseline / Full Pipeline)", "marker": "o"},
    "B": {"color": "#D97757", "label": "B (-SAR Ladder)", "marker": "s"},
    "C": {"color": "#9b59b6", "label": "C (-Portfolio Directive)", "marker": "d"},
    "D": {"color": "#2ecc71", "label": "D (-SMILES Shuffle)", "marker": "v"},
    "E": {"color": "#f1c40f", "label": "E (-Mechanism)", "marker": "<"},
    "F": {"color": "#e67e22", "label": "F (-Chemical Info / Blackout)", "marker": ">"},
    "G": {"color": "#a0522d", "label": "G (-Chem Blackout & -Portfolio / C+F)", "marker": "p"},
    "H": {"color": "#E74C3C", "label": "H (Full Ablation / All Combined)", "marker": "x"}
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


def plot_ablation_comparison(ablation_results, model_name, steps_count=14, output_path=None):
    """Plot the ablation-mode convergence results and save an output PNG."""
    if output_path is None:
        output_path = "output/generative_active_learning_ablation_comparison.png" if len(ablation_results) != 1 else f"output/generative_active_learning_ablation_{next(iter(ablation_results))}.png"

    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    steps = np.arange(1, steps_count + 1)

    # Sort available modes logically A to H
    sorted_modes = [m for m in AVAILABLE_ABLATION_MODES if m in ablation_results]

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
        except Exception:
            continue

        config = STYLE_CONFIGS.get(mode, {"color": "#888888", "label": mode, "marker": "o"})
        label_with_success = f"{config['label']} ({success_count}/{total_runs} hit 89%)"
        
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
    
    ax.set_title(f"Active Learning Ablation Analysis: {model_name.split('/')[-1]}", fontsize=15, fontweight="bold", pad=15)
    
    # Position the legend beautifully inside the bottom right of the plot area
    ax.legend(loc="lower right", fontsize=10.5, frameon=True, facecolor="white", edgecolor="none", shadow=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n-> Ablation plot successfully saved to '{output_path}'.")
    plt.close()
