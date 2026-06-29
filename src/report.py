import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import ALL_MODELS, FRONTIER_MODELS
import numpy as np
from src.paths import out_path
from src.prompts import ALL_LEVEL_KEYS

# --- Shared publication style -------------------------------------------------
# A single theme applied across all Results figures so the benchmark plots match
# the prospective-case plots (fonts, grid, DPI). Charts keep their own chart type;
# only the rendering is unified.
PALETTE_CONTEXT = ["#3477eb", "#e8833a"]  # with-context / without-context
PALETTE_MODELS = ["#3477eb", "#e8833a", "#3aa35a", "#9b59b6", "#e74c3c",
                  "#16a085", "#f1c40f", "#34495e"]

# Premium GPQA / Nature Chemistry style pastel/soft palette (from the reference image)
PALETTE_PREMIUM = [
    "#9A95D7",  # Soft Lavender/Purple
    "#7FBBE3",  # Soft Blue
    "#76DD9D",  # Soft Mint Green
    "#FA9F75",  # Soft Peach/Orange
    "#DE8285",  # Soft Salmon/Pink
    "#45CBD2",  # Soft Teal
    "#ED9510",  # Soft Amber/Gold
    "#9E758F",  # Soft Dusty Rose/Plum
]


def set_publication_style():
    """Apply a consistent, thesis-quality matplotlib theme."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.edgecolor": "#444444",
    })

def _clip_errorbars_to_ymax(ax, ymax):
    """Clamp any error-bar/whisker that overshoots ``ymax`` down to ``ymax``.

    Seaborn draws bar error bars as Line2D objects; we clip their y-data so a large
    SD never extends past the top of the axis (the whisker/cap stops exactly at max).
    """
    for line in ax.lines:
        ydata = line.get_ydata()
        if ydata is None or len(ydata) == 0:
            continue
        clipped = np.minimum(np.asarray(ydata, dtype=float), ymax)
        line.set_ydata(clipped)


def _task_num_catalysts(task_key):
    """Number of catalysts N in a Benchmark 2 task (for the random-guess baseline).

    Read from the ligand task definitions; returns None if unavailable so the
    caller simply omits the baseline rather than crashing.
    """
    try:
        from src.ligand_data import RANKING_TASKS
        task = RANKING_TASKS.get(task_key, {})
        return task.get("num_catalysts") or len(task.get("true_ranking", [])) or None
    except Exception:
        return None


def generate_assessment_report(df, filename=None):
    """Generates a Markdown file outlining the best, worst, and all comments per reaction."""
    if filename is None:
        filename = out_path("benchmark1", "assessment_highlights.md")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# LLM Chemistry Benchmark: Judge Assessments\n\n")
        
        for rxn in df['Reaction'].unique():
            rxn_df = df[df['Reaction'] == rxn].reset_index(drop=True)
            f.write(f"## Reaction: {rxn}\n\n")
            
            best_idx = rxn_df['Score'].idxmax()
            worst_idx = rxn_df['Score'].idxmin()
            
            best_row = rxn_df.iloc[best_idx]
            worst_row = rxn_df.iloc[worst_idx]
            
            f.write("### 🏆 Best Assessment\n")
            f.write(f"**Model:** {best_row['Model']} | **Context:** {best_row['Context']} | **Score:** {best_row['Score']}/10\n\n")
            f.write(f"> {best_row['Justification']}\n\n")
            
            f.write("### ⚠️ Worst Assessment\n")
            f.write(f"**Model:** {worst_row['Model']} | **Context:** {worst_row['Context']} | **Score:** {worst_row['Score']}/10\n\n")
            f.write(f"> {worst_row['Justification']}\n\n")
            
            f.write("### 📝 All Comments (Highlights)\n")
            for _, row in rxn_df.iterrows():
                f.write(f"* **{row['Model']}** (Score: {row['Score']}, {row['Context']}): {row['Justification']}\n")
            f.write("\n---\n\n")
            
    print(f"\n[+] Assessment highlights saved to {filename}")

def plot_fair_results(csv_path="output/fair_chemistry_results.csv"):
    """Generates and saves publication-style mean+/-SD bar charts for Benchmark 1.

    Scores saturate at 10 for strong models, so a boxplot collapses to invisible
    lines; a grouped barplot of the mean with SD error bars (n=5) reads cleanly and
    matches the captions. Frontier and open-weights models are visually separated.
    """
    set_publication_style()
    df = pd.read_csv(csv_path)
    model_order = [m.split('/')[-1] for m in ALL_MODELS]
    context_order = ["With Context", "Without Context"]

    for rxn in df['Reaction'].unique():
        rxn_df = df[df['Reaction'] == rxn]
        fig, ax = plt.subplots(figsize=(13, 7))

        # Grouped mean+/-SD bars (error bars = sample SD over the 5 replicates).
        sns.barplot(
            data=rxn_df, x='Model', y='Score', hue='Context',
            order=model_order, hue_order=context_order,
            palette=PALETTE_CONTEXT, errorbar='sd', capsize=0.12,
            err_kws={"linewidth": 1.5, "color": "#333333"}, ax=ax,
        )
        # Strip overlay so individual replicates stand out clearly over the bars.
        sns.stripplot(
            data=rxn_df, x='Model', y='Score', hue='Context',
            order=model_order, hue_order=context_order,
            dodge=True, jitter=0.12, marker='o', size=7, alpha=0.9,
            palette=['#12243f', '#5e3210'], edgecolor='white', linewidth=0.7,
            ax=ax, legend=False,
        )

        # Separator between proprietary frontier and open-weights models.
        ax.axvline(x=len(FRONTIER_MODELS) - 0.5, color='black', lw=1.5, linestyle=':')
        ymax = 10.6
        ax.text(len(FRONTIER_MODELS) / 2 - 0.5, ymax, "Proprietary frontier",
                ha='center', va='bottom', fontsize=11, style='italic', color='#555')
        ax.text((len(FRONTIER_MODELS) + len(model_order)) / 2 - 0.5, ymax, "Open-weights",
                ha='center', va='bottom', fontsize=11, style='italic', color='#555')

        # Clip SD whiskers so they never exceed the score scale max (10).
        _clip_errorbars_to_ymax(ax, 10)

        ax.set_ylim(0, 11)
        ax.set_ylabel("Judge Accuracy Score (0--10)")
        ax.set_xlabel("Model")
        ax.tick_params(axis='x', rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')

        h, l = ax.get_legend_handles_labels()
        ax.legend(h[:2], l[:2], title="Literature context", loc='best', framealpha=0.9)

        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(out_path("benchmark1", f"fair_plot_{rxn}.png"))
        plt.close(fig)
    print("[+] Benchmark 1 plots saved to output/benchmark1/.")
    plot_fair_star_results(csv_path)

def clean_display_name(name):
    """Clean and shorten name for clean sub-panel display."""
    name = name.replace("_", " ")
    name = name.replace("Morita-Baylis-Hillman", "MBH")
    name = name.replace("Pd-catalyzed", "Pd-Catalyzed")
    name = name.replace("Copper-catalyzed", "Cu-Catalyzed")
    if len(name) > 45:
        words = name.split(" ")
        mid = len(words) // 2
        name = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    return name

def plot_fair_star_results(csv_path="output/fair_chemistry_results.csv"):
    """Generates and saves a single, extremely beautiful, publication-quality star plot for Benchmark 1.

    Following the GPQA / Nature Chemistry paper style:
    - 1 single star plot (radar chart).
    - Spokes correspond to the 8 models (clean regular octagon).
    - Polygons correspond to the 6 reactions (Without Context scores only).
    - Grid lines are regular concentric octagons, not circular.
    - Outer boundary is a clean, dark, bold regular octagon.
    - Colors are colorblind-safe and modern (Okabe-Ito). No markers, no legends, no titles.
    """
    if not os.path.exists(csv_path):
        print(f"[!] Warning: Star plotting failed. File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    model_order = [m.split('/')[-1] for m in ALL_MODELS]
    present_models = [m for m in model_order if m in df['Model'].unique()]
    if not present_models:
        return

    N = len(present_models)
    if N == 0:
        return

    reactions = list(df['Reaction'].unique())

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Disable default polar grid lines and boundary circles
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('none')

    # Draw solid white background inside the regular polygon so it pops beautifully on grey slides/backgrounds
    ax.fill(angles, [10] * (N + 1), color='#ffffff', zorder=0)

    # Draw regular concentric polygons (grid lines) at 2, 4, 6, 8, 10
    grid_radii = [2, 4, 6, 8, 10]
    for r in grid_radii:
        ax.plot(angles, [r] * (N + 1), color='#e3e3e3', linestyle='-', linewidth=0.8, zorder=1)

    # Draw spokes
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 10], color='#e3e3e3', linestyle='-', linewidth=0.8, zorder=1)

    # Draw outer dark regular polygon boundary with increased thickness (bold contour)
    ax.plot(angles, [10] * (N + 1), color='#333333', linewidth=4.5, zorder=2)

    # Hide default ticks and tick labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Plot the "Without Context" scores for each reaction
    # Jewel-toned, high-contrast palette to maximize visibility and clarity of overlaps on grey/white
    PALETTE_JEWEL = [
        "#00529B",  # Deep Royal Blue
        "#D81B60",  # Vibrant Rose/Magenta
        "#00897B",  # Teal Green
        "#F4511E",  # Rich Amber Orange
        "#7B1FA2",  # Deep Purple
        "#43A047",  # Vibrant Forest Green
    ]

    for idx, rxn in enumerate(reactions):
        color = PALETTE_JEWEL[idx % len(PALETTE_JEWEL)]
        rxn_df = df[df['Reaction'] == rxn]
        values = []
        for model in present_models:
            sub = rxn_df[(rxn_df['Model'] == model) & (rxn_df['Context'] == 'Without Context')]
            val = sub['Score'].mean() if len(sub) > 0 else 0
            values.append(val if not np.isnan(val) else 0)
        values += values[:1]

        # Heavy-duty solid outlines with gorgeous, highly readable translucent fills
        ax.plot(angles, values, color=color, linewidth=4.2, zorder=4)
        ax.fill(angles, values, color=color, alpha=0.18, zorder=4)

    fig.tight_layout()
    fig.savefig(out_path("benchmark1", "fair_star_plots_combined.png"), dpi=300, transparent=True)
    fig.savefig(out_path("benchmark1", "fair_star_plots_combined.svg"), format='svg', transparent=True)
    plt.close(fig)
    print("[+] Benchmark 1 combined star plot saved to output/benchmark1/fair_star_plots_combined.png and .svg")

# --- BENCHMARK 2: LIGAND RANKING PLOTS ----------------------------------------

def plot_level_hierarchy(csv_path="output/benchmark2_results.csv"):
    if not os.path.exists(csv_path): 
        print(f"[!] Warning: Plotting failed. File not found: {csv_path}")
        return
        
    set_publication_style()
    df = pd.read_csv(csv_path)
    if df.empty: return

    # Only keep models actually present in the data, in config order, so a stale
    # config entry never reserves an empty slot (and real data is never dropped).
    model_order = [m.split('/')[-1] for m in ALL_MODELS
                   if m.split('/')[-1] in set(df['Model'].unique())]
    level_order = [lv for lv in ALL_LEVEL_KEYS if lv in set(df['Level'].unique())]
    model_palette = PALETTE_MODELS[:len(model_order)] if model_order else PALETTE_MODELS

    # Generate one plot per chemical task (e.g., Pd_Fluorination, Ni_Epoxide_Coupling)
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]
        fig, ax = plt.subplots(figsize=(13, 7))

        # Restrict hues to the models actually present for THIS task, so the legend
        # never advertises a model that has no bars in the plot.
        task_models = [m for m in model_order if m in set(task_df['Model'].unique())]
        task_palette = PALETTE_MODELS[:len(task_models)]

        # Mean +/- SD bars over the replicate iterations...
        sns.barplot(
            data=task_df, x='Level', y='Top_5_Overlap', hue='Model',
            order=level_order, hue_order=task_models, palette=task_palette,
            errorbar='sd', capsize=0.08, err_kws={"linewidth": 1.2, "color": "#333333"},
            alpha=0.55, ax=ax,
        )
        # ...with every individual replicate overlaid as a prominent jittered dot.
        sns.stripplot(
            data=task_df, x='Level', y='Top_5_Overlap', hue='Model',
            order=level_order, hue_order=task_models, palette=task_palette,
            dodge=True, jitter=0.15, marker='o', size=7, alpha=0.95,
            edgecolor='white', linewidth=0.7, ax=ax, legend=False,
        )

        # Overlap depth (k) may differ per task (e.g. Pd_Dual uses top-10).
        k = int(task_df['Top_K'].iloc[0]) if 'Top_K' in task_df.columns else 5
        # Clip SD whiskers so they never exceed the overlap scale max (k).
        _clip_errorbars_to_ymax(ax, k)

        # Random-guess baseline: the expected Top-k overlap if a model ranked the
        # N catalysts at random is k^2 / N (each of the k true-positives has a k/N
        # chance of landing in the predicted top-k). Drawn as a reference line so a
        # bar's height is read against "better than chance", not against zero.
        n_catalysts = _task_num_catalysts(task)
        if n_catalysts:
            chance = k * k / n_catalysts
            ax.axhline(chance, color="#c0392b", linestyle="--", linewidth=1.6, zorder=1)
            ax.text(0.995, chance, f" random guess ({chance:.2f})",
                    color="#c0392b", fontsize=11, va="bottom", ha="right",
                    transform=ax.get_yaxis_transform())

        ax.set_ylabel(f"Top-{k} Overlap (0--{k})")
        ax.set_xlabel("Prompting Complexity Level")
        ax.set_ylim(-0.3, k + 0.5)
        ax.tick_params(axis='x', rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')

        # De-duplicate legend (barplot + stripplot would double the handles) and
        # keep only the models actually plotted; place it inside the axes.
        h, l = ax.get_legend_handles_labels()
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(out_path("benchmark2", f"benchmark2_trend_{task}.png"))
        plt.close(fig)

    print("[+] Benchmark 2 plots saved to output/benchmark2/.")
    plot_level_hierarchy_star(csv_path)
    plt.close()

def format_level_label(label):
    """Keep only the short 'Level X' part for a clean star plot axis."""
    return label.split('(')[0].strip()

def plot_level_hierarchy_star(csv_path="output/benchmark2_results.csv"):
    """Generates and saves a single, extremely beautiful, publication-quality star plot for Benchmark 2.

    Following the GPQA / Nature Chemistry paper style:
    - 1 single star plot (radar chart).
    - Spokes correspond to the 6 prompting-complexity levels (Level 1 to Level 6).
    - Polygons correspond to the 4 ranking tasks.
    - Tasks are renormalized to a 0--1 fraction so every task shares the same radial scale.
    - Grid lines are regular concentric hexagons, not circular.
    - Outer boundary is a clean, dark, bold regular hexagon.
    - Colors are colorblind-safe and modern (Okabe-Ito). No markers, no legends, no titles.
    """
    if not os.path.exists(csv_path):
        print(f"[!] Warning: Star plotting failed. File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    level_order = [lv for lv in ALL_LEVEL_KEYS if lv in set(df['Level'].unique())]
    if not level_order:
        return

    categories = level_order
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    tasks = list(df['Task'].unique())

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Disable default polar grid lines and boundary circles
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('none')

    # Draw solid white background inside the regular polygon so it pops beautifully on grey slides/backgrounds
    ax.fill(angles, [1.0] * (N + 1), color='#ffffff', zorder=0)

    # Draw regular concentric polygons (grid lines) at 0.2, 0.4, 0.6, 0.8, 1.0
    grid_radii = [0.2, 0.4, 0.6, 0.8, 1.0]
    for r in grid_radii:
        ax.plot(angles, [r] * (N + 1), color='#e3e3e3', linestyle='-', linewidth=0.8, zorder=1)

    # Draw spokes
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1.0], color='#e3e3e3', linestyle='-', linewidth=0.8, zorder=1)

    # Draw outer dark regular polygon boundary with increased thickness (bold contour)
    ax.plot(angles, [1.0] * (N + 1), color='#333333', linewidth=4.5, zorder=2)

    # Hide all Level labels completely
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Jewel-toned, high-contrast palette to maximize visibility and clarity of overlaps on grey/white
    PALETTE_JEWEL = [
        "#00529B",  # Deep Royal Blue
        "#D81B60",  # Vibrant Rose/Magenta
        "#00897B",  # Teal Green
        "#F4511E",  # Rich Amber Orange
        "#7B1FA2",  # Deep Purple
        "#43A047",  # Vibrant Forest Green
    ]

    for idx, task in enumerate(tasks):
        color = PALETTE_JEWEL[idx % len(PALETTE_JEWEL)]
        task_df = df[df['Task'] == task]
        k = int(task_df['Top_K'].iloc[0]) if 'Top_K' in task_df.columns else 5
        values = []
        low_values = []
        high_values = []
        for lv in categories:
            sub = task_df[task_df['Level'] == lv]
            vals = sub['Top_5_Overlap'] if len(sub) > 0 else pd.Series([])
            mean_val = vals.mean() if len(vals) > 0 else 0
            sd_val = vals.std() if len(vals) > 1 else 0

            mean_val = mean_val if not np.isnan(mean_val) else 0
            sd_val = sd_val if not np.isnan(sd_val) else 0

            mean_frac = mean_val / k if k else 0
            sd_frac = sd_val / k if k else 0

            values.append(mean_frac)
            low_values.append(max(0.0, mean_frac - sd_frac))
            high_values.append(min(1.0, mean_frac + sd_frac))

        values += values[:1]
        low_values += low_values[:1]
        high_values += high_values[:1]

        # Heavy-duty solid outlines with gorgeous, highly readable translucent fills
        ax.plot(angles, values, linewidth=4.2, color=color, zorder=4)
        ax.fill(angles, values, color=color, alpha=0.18, zorder=4)

        # Plot the standard deviation/uncertainty bands around each task series
        ax.fill_between(angles, low_values, high_values, color=color, alpha=0.07, zorder=3)

    # Plot the random-guess baselines as dashed polygons
    # Each task's expected random overlap fraction is calculated as: random_fraction = k / N_catalysts.
    # We will average the random guess fraction over all tasks to draw a clean baseline polygon.
    random_fractions = []
    for task in tasks:
        n_cats = _task_num_catalysts(task)
        if n_cats:
            # For each task, expected random overlap is k^2/N. As a fraction of k, this is k/N.
            task_df = df[df['Task'] == task]
            k = int(task_df['Top_K'].iloc[0]) if 'Top_K' in task_df.columns else 5
            random_fractions.append(k / n_cats)
    
    if random_fractions:
        avg_random_fraction = np.mean(random_fractions)
        baseline_values = [avg_random_fraction] * (N + 1)
        # Draw a beautiful dashed red regular polygon as the random guess baseline
        ax.plot(angles, baseline_values, color="#c0392b", linestyle="--", linewidth=2.0, zorder=3)

    fig.tight_layout()
    fig.savefig(out_path("benchmark2", "benchmark2_star_plots_combined.png"), dpi=300, transparent=True)
    fig.savefig(out_path("benchmark2", "benchmark2_star_plots_combined.svg"), format='svg', transparent=True)
    plt.close(fig)
    print("[+] Benchmark 2 combined star plots saved to output/benchmark2/benchmark2_star_plots_combined.png and .svg")