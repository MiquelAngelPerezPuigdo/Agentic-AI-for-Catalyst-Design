import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import ALL_MODELS, FRONTIER_MODELS
import numpy as np
from src.paths import out_path

# --- Shared publication style -------------------------------------------------
# A single theme applied across all Results figures so the benchmark plots match
# the prospective-case plots (fonts, grid, DPI). Charts keep their own chart type;
# only the rendering is unified.
PALETTE_CONTEXT = ["#3477eb", "#e8833a"]  # with-context / without-context
PALETTE_MODELS = ["#3477eb", "#e8833a", "#3aa35a", "#9b59b6", "#e74c3c",
                  "#16a085", "#f1c40f", "#34495e"]


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

# --- ADDED FOR BENCHMARK 2: LIGAND RANKING PLOTS ---
from src.prompts import ALL_LEVEL_KEYS

def plot_level_hierarchy(csv_path="output/benchmark2_results.csv"):
    import os
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

        ax.set_ylabel(f"Top-{k} Overlap (0--{k})")
        ax.set_xlabel("Prompting Complexity Level")
        ax.set_ylim(-0.3, k + 0.5)
        ax.tick_params(axis='x', rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')

        # De-duplicate legend (barplot + stripplot would double the handles) and
        # keep only the models actually plotted; place it inside the axes.
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[:len(task_models)], l[:len(task_models)], title="Model",
                  loc='best', framealpha=0.9)

        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(out_path("benchmark2", f"benchmark2_trend_{task}.png"))
        plt.close(fig)

    print("[+] Benchmark 2 plots saved to output/benchmark2/.")
    plt.close()