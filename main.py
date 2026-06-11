import os
import argparse
from pathlib import Path

# Data Imports
from config import REACTIONS
from src.ligand_data import RANKING_TASKS

# Tool Imports
from src.document import extract_pdf_text_clean
from src.runner import run_fair_experiment, run_ligand_experiment
from src.report import plot_fair_results, generate_assessment_report, plot_level_hierarchy

def main():
    # Setup Terminal Arguments (The Master Switchboard)
    parser = argparse.ArgumentParser(description="LLM Chemistry Benchmark Suite")
    parser.add_argument("--mode", choices=["benchmark1", "benchmark2", "prospective"], 
                        required=True, help="Select which module to run.")
    from src.ablation import AVAILABLE_ABLATION_MODES
    parser.add_argument("--ablation", choices=["all", *AVAILABLE_ABLATION_MODES], default="A",
                        help="Select prospective ablation mode (A=Baseline, B=-SAR Ladder, C=-Portfolio Directive, D=-SMILES Shuffle, E=-Mechanism, F=-Chemical Blackout, G=-Chem Blackout & -Portfolio (C+F), H=Full Ablation, or 'all').")
    parser.add_argument("--model", type=str, default="google/gemini-3.5-flash",
                        help="Select model to run for prospective case.")
    parser.add_argument("--campaigns", type=int, default=5,
                        help="Number of active learning campaigns (default: 5).")
    parser.add_argument("--steps", type=int, default=14,
                        help="Number of prospective optimization steps / iterations (default: 14).")
    parser.add_argument("--save-details", action="store_true",
                        help="Save complete LLM justification, smiles, and step-by-step progress details.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # ==========================================
    # BENCHMARK 1: MECHANISTIC UNDERSTANDING
    # ==========================================
    if args.mode == "benchmark1":
        print("Initializing Benchmark 1: Mechanistic Understanding...")
        print("Extracting high-quality text context for all models...")
        
        for key in REACTIONS:
            p = Path(REACTIONS[key]["file"])
            if p.exists():
                REACTIONS[key]["text"] = extract_pdf_text_clean(p)
            else:
                print(f"[!] Warning: PDF not found at {p}")
                REACTIONS[key]["text"] = ""
            REACTIONS[key]["ground_truth"] = ""

        results_df = run_fair_experiment(REACTIONS, max_workers=5)
        
        csv_path = "output/fair_chemistry_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n[+] Raw data saved to {csv_path}")

        generate_assessment_report(results_df, "output/assessment_highlights.md")
        plot_fair_results(csv_path)
        print("\nBenchmark 1 complete! Check the 'output/' folder.")

    # ==========================================
    # BENCHMARK 2: LIGAND RANKING
    # ==========================================
    elif args.mode == "benchmark2":
        print("Initializing Benchmark 2: Ligand Ranking...")
        print("Extracting contextual paper texts...")
        
        # Load the text from the PDFs directly into your RANKING_TASKS dictionary
        for key, task in RANKING_TASKS.items():
            task["related_paper_text"] = "\n".join([extract_pdf_text_clean(Path(p)) for p in task["related_paper_paths"] if Path(p).exists()])
            task["actual_paper_text"] = "\n".join([extract_pdf_text_clean(Path(p)) for p in task["actual_paper_paths"] if Path(p).exists()])

        results_df = run_ligand_experiment(max_workers=5) 
        
        csv_path = "output/benchmark2_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n[+] Raw data saved to {csv_path}")
        
        plot_level_hierarchy(csv_path)
        print("\nBenchmark 2 complete! Check the 'output/' folder.")
        
   # ==========================================
    # PROSPECTIVE CASE
    # ==========================================
    elif args.mode == "prospective":
        print(f"\nInitializing Prospective Active Learning Case with Ablation Mode {args.ablation}...")
        from src.ablation import run_all_ablations_experiment
        
        output_path = run_all_ablations_experiment(
            model=args.model,
            campaigns=args.campaigns,
            steps=args.steps,
            target_mode=args.ablation,
            save_details=args.save_details
        )
        print(f"\nProspective Case (Ablation {args.ablation}) complete! Plot saved to {output_path}.")

if __name__ == "__main__":
    main()

