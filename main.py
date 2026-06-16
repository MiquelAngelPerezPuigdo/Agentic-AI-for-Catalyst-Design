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
from src.paths import out_path

def main():
    # Setup Terminal Arguments (The Master Switchboard)
    parser = argparse.ArgumentParser(description="LLM Chemistry Benchmark Suite")
    parser.add_argument("--mode", choices=["benchmark1", "benchmark2", "prospective", "saturn-mbh"], 
                        required=True, help="Select which module to run.")
    from src.ablation import AVAILABLE_ABLATION_MODES
    parser.add_argument("--ablation", choices=["all", *AVAILABLE_ABLATION_MODES], default="A",
                        help="Select prospective ablation mode (A=Baseline (Shuffled SMILES), B=-SAR Ladder, C=-SMILES Shuffle, D=-Mechanism, E=-Chemical Blackout, F=Full Ablation, or 'all').")
    parser.add_argument("--model", type=str, default="google/gemini-3.5-flash",
                        help="Select model to run for prospective case.")
    parser.add_argument("--campaigns", type=int, default=5,
                        help="Number of active learning campaigns (default: 5).")
    parser.add_argument("--steps", type=int, default=14,
                        help="Number of prospective optimization steps / iterations (default: 14).")
    parser.add_argument("--constraint", choices=["batch", "divergent", "click"], default=None,
                        help="Run a Real-World Constrained Campaign instead of an ablation (batch=6x7 high-throughput, divergent=shared-scaffold synthesis with RDKit MCS audit, click=SDL combinatorial click-library search over building blocks).")
    parser.add_argument("--chem-agnostic", action="store_true",
                        help="For --constraint click: strip all reaction context/mechanism from the prompt (analog of Ablation E) to test if chemistry knowledge helps library selection.")
    parser.add_argument("--save-details", action="store_true",
                        help="Save complete LLM justification, smiles, and step-by-step progress details.")
    # --- saturn-mbh mode: de novo MBH catalyst design via an external Saturn install ---
    parser.add_argument("--saturn-home", type=str, default=None,
                        help="Path to the external Saturn checkout (default: $SATURN_HOME or ~/saturn).")
    parser.add_argument("--saturn-env", type=str, default="saturn",
                        help="Conda env name in which to run Saturn (default: 'saturn').")
    parser.add_argument("--budget", type=int, default=500,
                        help="saturn-mbh: oracle-call budget for the campaign (default: 500).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="saturn-mbh: device passed to Saturn ('cuda' or 'cpu', default: 'cuda').")
    parser.add_argument("--dry-run", action="store_true",
                        help="saturn-mbh: inject oracle and write the config but do not launch Saturn.")
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
        
        csv_path = out_path("benchmark1", "fair_chemistry_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n[+] Raw data saved to {csv_path}")

        generate_assessment_report(results_df, out_path("benchmark1", "assessment_highlights.md"))
        plot_fair_results(csv_path)
        print("\nBenchmark 1 complete! Check the 'output/benchmark1/' folder.")

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
        
        csv_path = out_path("benchmark2", "benchmark2_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n[+] Raw data saved to {csv_path}")
        
        plot_level_hierarchy(csv_path)
        print("\nBenchmark 2 complete! Check the 'output/benchmark2/' folder.")
        
   # ==========================================
    # PROSPECTIVE CASE
    # ==========================================
    elif args.mode == "prospective":
        # Ensure GOOGLE_API_KEY is active if we are running Gemini
        if "gemini" in args.model.lower():
            from config import GOOGLE_API_KEY
            if not GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == "":
                import sys
                sys.exit("\n[!] CRITICAL ERROR: GOOGLE_API_KEY is not set or empty, but a Gemini model was requested. Stopping campaign to prevent OpenRouter fallback or unexpected API costs!")

        # Branch: Real-World Constrained Campaign vs. standard ablation study
        if args.constraint == "click":
            variant = "Chemistry-Agnostic Click" if args.chem_agnostic else "SDL Click-Library"
            print(f"\nInitializing {variant} Campaign...")
            from src.ablation import run_click_experiment

            output_path = run_click_experiment(
                model=args.model,
                campaigns=args.campaigns,
                steps=args.steps,
                save_details=args.save_details,
                chem_agnostic=args.chem_agnostic
            )
            print(f"\n{variant} Campaign complete! Plot saved to {output_path}.")
        elif args.constraint:
            print(f"\nInitializing Real-World Constrained Campaign: '{args.constraint}'...")
            from src.ablation import run_constrained_experiment

            output_path = run_constrained_experiment(
                model=args.model,
                campaigns=args.campaigns,
                steps=args.steps,
                constraint=args.constraint,
                save_details=args.save_details
            )
            print(f"\nConstrained Campaign ('{args.constraint}') complete! Plot saved to {output_path}.")
        else:
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

    # ==========================================
    # SATURN-MBH: DE NOVO MBH CATALYST DESIGN
    # ==========================================
    elif args.mode == "saturn-mbh":
        print("Initializing Saturn-MBH de novo catalyst design campaign...")
        from src.saturn_mbh import run_mbh_campaign

        run_dir = run_mbh_campaign(
            model=args.model,
            budget=args.budget,
            saturn_home=args.saturn_home,
            saturn_env=args.saturn_env,
            device=args.device,
            dry_run=args.dry_run,
        )
        print(f"\nSaturn-MBH campaign artifacts in: {run_dir}")

if __name__ == "__main__":
    main()

