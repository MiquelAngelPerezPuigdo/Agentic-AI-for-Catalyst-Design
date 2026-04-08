import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import ALL_MODELS, FRONTIER_MODELS

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