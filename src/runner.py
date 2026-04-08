import concurrent.futures
import pandas as pd
from tqdm import tqdm
from src.api import call_openrouter
from src.evaluation import evaluate_response
from src.prompts import PROMPT_TEMPLATE
from config import ALL_MODELS, FRONTIER_MODELS, ITERATIONS, JUDGE_MODEL

def process_single_task(model, rxn_key, rxn_data, use_context, iteration):
    """Worker function to handle a single prompt and evaluation cycle."""
    if use_context:
        ctx = f"Context from research paper:\n{rxn_data['text'][:50000]}"
    else:
        ctx = "No specific context provided. Use general chemical knowledge."

    prompt = PROMPT_TEMPLATE.format(reaction_name=rxn_data["name"], context_block=ctx)
    
    # 1. Get model response
    resp = call_openrouter(model, prompt)
    
    # 2. Evaluate response
    score, just = evaluate_response(rxn_data["name"], rxn_data["ground_truth"], resp, use_context)
    
    # 3. Return the row dictionary
    return {
        "Model": model.split('/')[-1],
        "Reaction": rxn_key,
        "Context": "With Context" if use_context else "Without Context",
        "Score": score,
        "Type": "Proprietary" if model in FRONTIER_MODELS else "Non-Proprietary",
        "Model_Response": resp,       
        "Judge_Justification": just   
    }

def run_fair_experiment(reactions_dict, max_workers=5):
    """Orchestrates the parallel execution of the benchmark."""
    print("\nPhase 1: Generating Ground Truths...")
    for key, rxn in reactions_dict.items():
        truth_prompt = PROMPT_TEMPLATE.format(
            reaction_name=rxn["name"],
            context_block=f"Full Paper Text:\n{rxn['text'][:50000]}"
        )
        rxn["ground_truth"] = call_openrouter(JUDGE_MODEL, truth_prompt, temperature=0)

    results = []
    print(f"\nPhase 2: Benchmarking all models (Parallelized with {max_workers} workers)...")
    
    tasks = []
    for model in ALL_MODELS:
        for rxn_key, rxn_data in reactions_dict.items():
            for use_context in [True, False]:
                for i in range(ITERATIONS):
                    tasks.append((model, rxn_key, rxn_data, use_context, i))
                    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_task, *task_args): task_args 
            for task_args in tasks
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="API Calls"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"\n[!] A task generated an exception: {exc}")

    df = pd.DataFrame(results)
    return df