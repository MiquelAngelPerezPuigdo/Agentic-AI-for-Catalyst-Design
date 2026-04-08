import os
from pathlib import Path
from config import REACTIONS
from src.document import extract_pdf_text_clean
from src.runner import run_fair_experiment
from src.report import plot_fair_results, generate_assessment_report

def main():
    # Make sure output directory exists
    os.makedirs("output", exist_ok=True)
    
    print("Initializing LLM Chemistry Benchmark...")

    # 1. Prepare Data
    print("Extracting high-quality text context for all models...")
    for key in REACTIONS:
        p = Path(REACTIONS[key]["file"])
        if p.exists():
            REACTIONS[key]["text"] = extract_pdf_text_clean(p)
        else:
            print(f"[!] Warning: PDF not found at {p}")
            REACTIONS[key]["text"] = ""
        REACTIONS[key]["ground_truth"] = ""

    # 2. Run the Experiment
    # Adjust max_workers if you run into rate limits with OpenRouter
    results_df = run_fair_experiment(REACTIONS, max_workers=5)
    
    # Save raw CSV data
    csv_path = "output/fair_chemistry_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Raw data saved to {csv_path}")

    # 3. Generate Reports
    generate_assessment_report(results_df, "output/assessment_highlights.md")
    plot_fair_results(csv_path)
    
    print("\nBenchmark complete! Check the 'output/' folder for your files.")

if __name__ == "__main__":
    main()