import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import ALL_MODELS, FRONTIER_MODELS
import numpy as np

def generate_assessment_report(df, filename="output/assessment_highlights.md"):
    """Generates a Markdown file outlining the best, worst, and all comments per reaction."""
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
    """Generates and saves seaborn boxplots for the benchmark results."""
    df = pd.read_csv(csv_path)
    model_order = [m.split('/')[-1] for m in ALL_MODELS]
    
    for rxn in df['Reaction'].unique():
        plt.figure(figsize=(14, 8))
        rxn_df = df[df['Reaction'] == rxn]
        
        sns.boxplot(data=rxn_df, x='Model', y='Score', hue='Context', order=model_order,
                    palette=['#2ecc71', '#e67e22'], showfliers=False, boxprops=dict(alpha=.3))
        
        sns.stripplot(data=rxn_df, x='Model', y='Score', hue='Context', order=model_order,
                      dodge=True, jitter=True, marker='o', alpha=0.7, palette=['#27ae60', '#d35400'])
        
        plt.axvline(x=len(FRONTIER_MODELS)-0.5, color='black', lw=2, linestyle=':')
        plt.title(f"Fair Benchmark (Text-Only Context): {rxn}", fontsize=14)
        plt.ylim(-0.5, 11)
        plt.ylabel("Accuracy Score (0-10)")
        plt.xticks(rotation=15)
        
        h, l = plt.gca().get_legend_handles_labels()
        plt.legend(h[:2], l[:2], title="Condition", loc='lower left')
        
        plt.tight_layout()
        plt.savefig(f"output/fair_plot_{rxn}.png")

# --- ADDED FOR BENCHMARK 2: LIGAND RANKING PLOTS ---
from src.prompts import ALL_LEVEL_KEYS

def plot_level_hierarchy(csv_path="output/benchmark2_results.csv"):
    import os
    if not os.path.exists(csv_path): 
        print(f"[!] Warning: Plotting failed. File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    # Ensure consistent ordering
    model_order = [m.split('/')[-1] for m in ALL_MODELS]
    level_order = ALL_LEVEL_KEYS
    
    # Adaptive palettes based on the number of models in config.py
    colors_box = ['#2ecc71', '#e67e22', '#3498db', '#9b59b6', '#34495e']
    colors_strip = ['#27ae60', '#d35400', '#2980b9', '#8e44ad', '#2c3e50']
    box_palette = colors_box[:len(model_order)] if len(model_order) > 1 else [colors_box[0]]
    strip_palette = colors_strip[:len(model_order)] if len(model_order) > 1 else [colors_strip[0]]

    # Generate one plot per chemical task (e.g., Pd_Fluorination, Ni_Epoxide_Coupling)
    for task in df['Task'].unique():
        plt.figure(figsize=(13, 7))
        task_df = df[df['Task'] == task]
        
        # Draw Boxplots
        sns.boxplot(data=task_df, x='Level', y='Top_5_Overlap', hue='Model', 
                    order=level_order, hue_order=model_order, palette=box_palette,
                    showfliers=False, boxprops=dict(alpha=.3))
        
        # Draw Stripplots overlay
        sns.stripplot(data=task_df, x='Level', y='Top_5_Overlap', hue='Model', 
                      order=level_order, hue_order=model_order, palette=strip_palette,
                      dodge=True, jitter=True, marker='o', alpha=0.7)
        
        plt.title(f"Performance Analysis (Benchmark 2): {task}", fontsize=14)
        plt.ylabel("Top 5 Overlap Score (0-5)", fontsize=12)
        plt.xlabel("Prompting Complexity Level", fontsize=12)
        plt.xticks(rotation=20, ha='right')
        plt.ylim(-0.5, 5.5) 
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        # Save output
        plt.savefig(f"output/benchmark2_trend_{task}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"[+] Benchmark 2 plots successfully saved to the output/ directory.")

    def plot_prospective_convergence(results_dict, output_path="output/generative_benchmark.png"):
    plt.figure(figsize=(11, 7))
    steps = np.arange(1, 15) # 14 Iterations
    
    # Custom colors mapping for specific models if they are used
    colors = {"google/gemini-3.1-pro-preview": "#4285F4", "anthropic/claude-opus-4.8": "#D97757", "openai/gpt-5.5": "#6A737D"}
    fallback_colors = ["#9b59b6", "#2ecc71", "#f1c40f", "#e74c3c"]
    
    color_idx = 0
    for model, runs in results_dict.items():
        if not runs: continue
        arr = np.array(runs)
        mean_yield = np.mean(arr, axis=0)
        std_yield = np.std(arr, axis=0)
        
        c = colors.get(model, fallback_colors[color_idx % len(fallback_colors)])
        color_idx += 1
        model_label = model.split('/')[-1]
        
        plt.plot(steps, mean_yield, color=c, linestyle="-", marker="o", linewidth=2.5, label=model_label)
        plt.fill_between(steps, mean_yield - std_yield, mean_yield + std_yield, color=c, alpha=0.2)
        
        for run in runs:
            plt.plot(steps, run, color=c, linestyle="--", alpha=0.15, linewidth=1)

    plt.axhline(y=89.0, color="#2ca02c", linestyle=":", linewidth=2, label="Best Database Hit (89.0% Limit)")
    
    plt.xlabel("Active Learning Optimization Step", fontsize=12)
    plt.ylabel("Maximum Discovered Yield (%)", fontsize=12)
    plt.xticks(steps)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"-> Active Learning convergence plot saved to '{output_path}'.")
    plt.close()