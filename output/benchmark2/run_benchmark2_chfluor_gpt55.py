#!/usr/bin/env python
"""
Custom benchmark2 runner: CHFluorination task only with ChatGPT-5.5
"""
import os
from pathlib import Path

# Data Imports
from src.ligand_data import RANKING_TASKS
from src.document import extract_pdf_text_clean
from src.runner import run_ligand_experiment
from src.report import plot_level_hierarchy
from src.paths import out_path
import pandas as pd
import concurrent.futures
from src.prompts import PROMPTS_TEXT_MODE, PROMPTS_JSON_MODE, SYSTEM_PROMPT_JSON, ALL_LEVEL_KEYS
from src.api import call_openrouter
from src.evaluation import extract_scores_text, extract_scores_json, calculate_top_5_overlap
from tqdm import tqdm

# Override models and iterations
original_all_models = __import__('config').ALL_MODELS
original_iterations = __import__('config').ITERATIONS

def run_chfluorination_benchmark():
    """Run benchmark2 for CHFluorination task only with GPT-5.5"""
    
    print("=" * 80)
    print("CUSTOM BENCHMARK 2: CHFluorination (Pd_CH_Fluorination) with ChatGPT-5.5")
    print("=" * 80)
    
    # Override with ChatGPT-5.5 only
    test_models = ["openai/gpt-5.5"]
    
    # Extract contextual paper texts for CHFluorination
    print("\n[1] Extracting contextual paper texts for Pd_CH_Fluorination...")
    task_key = "Pd_CH_Fluorination"
    task = RANKING_TASKS[task_key]
    
    task["related_paper_text"] = "\n".join([
        extract_pdf_text_clean(Path(p)) 
        for p in task["related_paper_paths"] if Path(p).exists()
    ])
    task["actual_paper_text"] = "\n".join([
        extract_pdf_text_clean(Path(p)) 
        for p in task["actual_paper_paths"] if Path(p).exists()
    ])
    print(f"[+] Extracted texts for {task_key}")
    
    # Prepare task data for all levels
    print("\n[2] Preparing task data for all ranking levels...")
    all_levels = ALL_LEVEL_KEYS
    task_data_by_level = {}
    
    for level in all_levels:
        task_data = {
            "reaction_definition": task["reaction_definition"],
            "catalyst_list": task["catalyst_list"],
            "id_map": task["id_map"],
            "true_ranking": task["true_ranking"],
            "num_catalysts": task["num_catalysts"],
            "mechanism_text": task["mechanism_text"],
            "related_paper_text": task["related_paper_text"],
            "actual_paper_text": task["actual_paper_text"]
        }
        task_data_by_level[level] = task_data
    
    # Run the experiment with GPT-5.5 only
    print(f"\n[3] Running Ligand Ranking Benchmark with {len(test_models)} model(s) and {len(all_levels)} levels...")
    print(f"    Models: {test_models}")
    print(f"    Task: {task_key}")
    print(f"    Levels: {all_levels}\n")
    
    results = []
    iterations = 5
    max_workers = 3
    
    # Build all tasks (model, task, level, iteration combinations)
    all_tasks = [
        (model, task_key, level, iteration, task_data_by_level[level])
        for model in test_models
        for level in all_levels
        for iteration in range(iterations)
    ]
    
    print(f"Total API calls to make: {len(all_tasks)}")
    print(f"Using {max_workers} workers for parallel execution\n")
    
    def process_task(model, task_key_arg, level_name, iteration, task_data):
        """Worker function to handle a single Ligand Ranking prompt and evaluation."""
        is_json_mode = level_name in PROMPTS_JSON_MODE
        
        # 1. Format the correct prompt based on the level's routing strategy
        if is_json_mode:
            system_prompt = SYSTEM_PROMPT_JSON.format(**task_data)
            user_prompt = PROMPTS_JSON_MODE[level_name].format(**task_data)
        else:
            full_prompt = PROMPTS_TEXT_MODE[level_name].format(**task_data)
            if "CRITICAL INSTRUCTIONS:" in full_prompt:
                user_prompt, system_prompt = full_prompt.split("CRITICAL INSTRUCTIONS:")
                system_prompt = "CRITICAL INSTRUCTIONS:\n" + system_prompt
            else:
                user_prompt = full_prompt
                system_prompt = "You are an expert computational chemist."

        # 2. Call the OpenRouter API
        response = call_openrouter(model, user_prompt, system_prompt=system_prompt)
        if response in ["API_RETURNED_NULL", "API_ERROR"] or not response: 
            return None

        # 3. Parse the scores based on the level mode
        if is_json_mode:
            predicted_scores_dict = extract_scores_json(response)
        else:
            predicted_scores_dict = extract_scores_text(response)

        # 4. Map the IDs back to SMILES and calculate the score
        ligand_score_pairs = [
            (task_data["id_map"][pid], float(score)) 
            for pid, score in predicted_scores_dict.items() 
            if pid in task_data["id_map"]
        ]
                
        # If the model failed to score EVERY catalyst, reject the run (Strict Evaluation)
        if len(ligand_score_pairs) != task_data["num_catalysts"]:
            return None 
                
        ligand_score_pairs.sort(key=lambda x: x[1], reverse=True)
        predicted_ranking = [pair[0] for pair in ligand_score_pairs]
        
        return {
            "Model": model.split('/')[-1],
            "Task": task_key_arg,
            "Level": level_name,
            "Iteration": iteration, 
            "Top_5_Overlap": calculate_top_5_overlap(predicted_ranking, task_data["true_ranking"]),
            "Valid_Ligands_Ranked": len(predicted_ranking)
        }
    
    # Execute all tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_task, *task_args): task_args
            for task_args in all_tasks
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_tasks), desc="Running Benchmark"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"\n[!] Task generated an exception: {exc}")
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    print(f"\n[+] Completed {len(results)} successful evaluations")
    print(f"\nResults Summary:")
    print(results_df.groupby(['Model', 'Level'])['Top_5_Overlap'].agg(['mean', 'std', 'count']))
    
    # Save results
    csv_path = out_path("benchmark2", "benchmark2_chfluorination_gpt55_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Results saved to: {csv_path}")
    
    return results_df

if __name__ == "__main__":
    results_df = run_chfluorination_benchmark()
