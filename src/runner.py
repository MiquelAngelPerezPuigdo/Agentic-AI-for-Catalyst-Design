import concurrent.futures
import pandas as pd
from tqdm import tqdm
from src.api import call_openrouter
from src.evaluation import evaluate_response
from src.prompts import PROMPT_TEMPLATE
from config import ALL_MODELS, FRONTIER_MODELS, ITERATIONS, JUDGE_MODEL
# --- ADDED FOR BENCHMARK 2: LIGAND RANKING ---
from src.ligand_data import RANKING_TASKS
from src.prompts import PROMPTS_TEXT_MODE, PROMPTS_JSON_MODE, SYSTEM_PROMPT_JSON, ALL_LEVEL_KEYS
from src.evaluation import extract_scores_text, extract_scores_json, calculate_top_5_overlap
import json
# --- ADDED FOR PROSPECTIVE CASE ---
import re
from src.ligand_data import INITIAL_SEED

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

def generate_single_ground_truth(rxn_name, rxn_text, max_retries=3):
    """Generates the ground truth assessment for a single reaction with retries."""
    truth_prompt = PROMPT_TEMPLATE.format(
        reaction_name=rxn_name,
        context_block=f"Full Paper Text:\n{rxn_text[:50000]}"
    )
    for attempt in range(max_retries):
        try:
            truth = call_openrouter(JUDGE_MODEL, truth_prompt, temperature=0)
            if truth and truth not in ["API_ERROR", "API_RETURNED_NULL"] and not truth.startswith("ERROR"):
                return truth
            print(f"[!] Warning: Ground truth generation failed on attempt {attempt+1}/{max_retries} for {rxn_name}. Retrying...")
        except Exception as e:
            print(f"[!] Warning: Exception during ground truth generation on attempt {attempt+1}/{max_retries} for {rxn_name}: {e}")
    raise RuntimeError(f"Failed to generate valid ground truth for {rxn_name} after {max_retries} attempts.")

def run_fair_experiment(reactions_dict, max_workers=5):
    """Orchestrates the parallel execution of the benchmark."""
    print("\nPhase 1: Generating Ground Truths in Parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(reactions_dict), max_workers)) as executor:
        future_to_rxn = {
            executor.submit(generate_single_ground_truth, rxn["name"], rxn.get("text", "")): key
            for key, rxn in reactions_dict.items()
        }
        for future in concurrent.futures.as_completed(future_to_rxn):
            key = future_to_rxn[future]
            try:
                reactions_dict[key]["ground_truth"] = future.result()
                print(f"[+] Successfully generated ground truth for: {reactions_dict[key]['name'][:60]}...")
            except Exception as exc:
                print(f"\n[!] Critical failure in ground truth generation for {key}: {exc}")
                raise exc

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

def process_ligand_task(model, task_key, level_name, iteration, task_data):
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

    # 2. Call the OpenRouter API (Reusing your clean API function!)
    # Note: We pass the system_prompt here, which requires a slight update to api.py later.
    response = call_openrouter(model, user_prompt, system_prompt=system_prompt)
    if response in ["API_RETURNED_NULL", "API_ERROR"] or not response: 
        return None

    # 3. Parse the scores based on the level mode
    if is_json_mode:
        predicted_scores_dict = extract_scores_json(response)
    else:
        predicted_scores_dict = extract_scores_text(response)

    # 4. Map the IDs back to SMILES and calculate the score
    ligand_score_pairs = [(task_data["id_map"][pid], float(score)) 
                          for pid, score in predicted_scores_dict.items() if pid in task_data["id_map"]]
            
    # If the model failed to score EVERY catalyst, reject the run (Strict Evaluation)
    if len(ligand_score_pairs) != task_data["num_catalysts"]:
        return None 
            
    ligand_score_pairs.sort(key=lambda x: x[1], reverse=True)
    predicted_ranking = [pair[0] for pair in ligand_score_pairs]
    
    return {
        "Model": model.split('/')[-1],
        "Task": task_key,
        "Level": level_name,
        "Iteration": iteration, 
        "Top_5_Overlap": calculate_top_5_overlap(predicted_ranking, task_data["true_ranking"]),
        "Valid_Ligands_Ranked": len(predicted_ranking)
    }

def run_ligand_experiment(max_workers=5, required_iterations=5, max_attempts=20):
    """Orchestrates the parallel execution of the Ligand Benchmark with dynamic retry queueing."""
    results = []
    
    # Initialize trackers for each combination
    trackers = {}
    for model in ALL_MODELS:
        for task_key in RANKING_TASKS:
            for level_name in ALL_LEVEL_KEYS:
                trackers[(model, task_key, level_name)] = {
                    "successes_got": 0,
                    "attempts_submitted": 0,
                    "pending_attempts": 0
                }

    print(f"\nPhase 2: Benchmarking Ligand Ranking (Parallelized with {max_workers} workers)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        
        # 1. Submit initial tasks (exactly required_iterations per combination)
        for (model, task_key, level_name), tracker in trackers.items():
            task_data = RANKING_TASKS[task_key]
            for i in range(required_iterations):
                future = executor.submit(process_ligand_task, model, task_key, level_name, i, task_data)
                future_to_info[future] = (model, task_key, level_name, i, task_data)
                tracker["attempts_submitted"] += 1
                tracker["pending_attempts"] += 1

        # Use a progress bar for the total expected successes
        total_required_successes = len(trackers) * required_iterations
        pbar = tqdm(total=total_required_successes, desc="Ligand Successes")
        
        while future_to_info:
            # Wait for any future to complete
            done, _ = concurrent.futures.wait(future_to_info.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            
            for future in done:
                if future not in future_to_info:
                    continue
                model, task_key, level_name, attempt_idx, task_data = future_to_info.pop(future)
                tracker_key = (model, task_key, level_name)
                tracker = trackers[tracker_key]
                tracker["pending_attempts"] -= 1
                
                try:
                    res = future.result()
                except Exception as e:
                    print(f"\n[!] Exception during process_ligand_task for {model.split('/')[-1]} on {task_key}/{level_name}: {e}")
                    res = None
                
                if res:
                    # Successful run!
                    res["Iteration"] = tracker["successes_got"]
                    results.append(res)
                    tracker["successes_got"] += 1
                    pbar.update(1)
                
                # Dynamic adjustment check:
                # If we still need more successes, and the current pending queue isn't enough to reach required_iterations,
                # and we haven't hit the max_attempts cap, submit a new retry task on-demand.
                needed = required_iterations - tracker["successes_got"]
                if needed > tracker["pending_attempts"] and tracker["attempts_submitted"] < max_attempts:
                    next_attempt_idx = tracker["attempts_submitted"]
                    new_future = executor.submit(process_ligand_task, model, task_key, level_name, next_attempt_idx, task_data)
                    future_to_info[new_future] = (model, task_key, level_name, next_attempt_idx, task_data)
                    tracker["attempts_submitted"] += 1
                    tracker["pending_attempts"] += 1

        pbar.close()

    return pd.DataFrame(results)

def clean_json_markdown(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1)
    bracket_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if bracket_match: return bracket_match.group(0)
    return text

def run_prospective_campaign(campaign_id, model_name):
    # Only import heavy dependencies if this function is actually called
    try:
        from rdkit import Chem
        from src.surrogate import init_surrogate, score_ligand
        surrogate, fp_gen, dataset_dict = init_surrogate()
    except ImportError:
        return [] # Returns empty if RDkit isn't installed

    from src.prompts import PROSPECTIVE_EXPLORE_DIRECTIVE, PROSPECTIVE_EXPLOIT_DIRECTIVE, PROSPECTIVE_SYSTEM_PROMPT
    
    print(f"[{model_name.split('/')[-1]} | Campaign {campaign_id}] Starting...")
    history_records = []
    
    for line in INITIAL_SEED.strip().split("\n"):
        line = line.strip()
        if line:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                yield_val, smiles_val = parts[0].strip(), parts[1].strip()
                try:
                    mol_obj = Chem.MolFromSmiles(smiles_val)
                    smiles_to_store = Chem.MolToSmiles(mol_obj, canonical=True) if mol_obj else smiles_val
                except Exception: smiles_to_store = smiles_val
                history_records.append({"yield": yield_val, "smiles": smiles_to_store, "comment": None})

    TOTAL_ITERATIONS = 14
    max_yield_history = []
    current_global_max = 0.0

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        shuffled_history_lines = []
        for record in history_records:
            try:
                mol_obj = Chem.MolFromSmiles(record["smiles"])
                enumerated_smiles = Chem.MolToSmiles(mol_obj, doRandom=True) if mol_obj else record["smiles"]
            except Exception: enumerated_smiles = record["smiles"]
            shuffled_history_lines.append(f"{record['yield']}\t{enumerated_smiles}")
            
        history_text_for_llm = "\n".join(shuffled_history_lines)
        directive = PROSPECTIVE_EXPLORE_DIRECTIVE if iteration <= 7 else PROSPECTIVE_EXPLOIT_DIRECTIVE
        
        sys_prompt = PROSPECTIVE_SYSTEM_PROMPT.format(iteration=iteration, total_iterations=TOTAL_ITERATIONS, portfolio_directive=directive)
        user_prompt = f"Here is the history of tried ligands and their yields:\n\n{history_text_for_llm}\n\nThink carefully, output your justification, and provide your 3 next proposed ligands."
            
        max_gen_attempts = 5
        for attempt in range(max_gen_attempts):
            try:
                # Reuses your clean API block!
                response = call_openrouter(model_name, user_prompt, system_prompt=sys_prompt) 
                json_str = clean_json_markdown(response)
                proposed_ligands = json.loads(json_str.strip())
                smiles_list = [item['smiles'] for item in proposed_ligands]
                
                if len(smiles_list) != 3: raise ValueError("Did not generate exactly 3.")
                
                proposed_canonicals = []
                existing_canonicals = set([r["smiles"] for r in history_records])
                for s in smiles_list:
                    m = Chem.MolFromSmiles(s, sanitize=False)
                    if m is None: raise ValueError("Invalid SMILES")
                    Chem.SanitizeMol(m)
                    can_s = Chem.MolToSmiles(m, canonical=True)
                    if can_s in existing_canonicals or can_s in proposed_canonicals: raise ValueError("Duplicate SMILES")
                    proposed_canonicals.append(can_s)
                
                break # Success! Break out of retry loop.
            except Exception:
                if attempt == max_gen_attempts - 1:
                    import random
                    existing_canonicals = set([r["smiles"] for r in history_records])
                    untried_candidates = [smiles for smiles in dataset_dict if smiles not in existing_canonicals]
                    if len(untried_candidates) >= 3:
                        smiles_list = random.sample(untried_candidates, 3)
                    else:
                        hardcoded_defaults = ["C1=CC=C(C=C1)CN2C=C(C3=NC(=CC=C3)O)N=N2", "Oc1nc(C2=CN(CC3=CC=C(C=C3)C(F)(F)F)N=N2)c(C)cc1", "COC1=C(N=C(C=C1)C2=CN(CC3=CC=CC=C3)N=N2)O"]
                        smiles_list = [s for s in hardcoded_defaults if s not in existing_canonicals]
                        while len(smiles_list) < 3:
                            smiles_list.append(f"CCCC{len(smiles_list)}C")

        # Evaluate and log
        step_yields = []
        for smiles in smiles_list:
            chosen_yield, stored_smiles = score_ligand(smiles, surrogate, fp_gen, dataset_dict, history_records)
            step_yields.append(chosen_yield)
            history_records.append({"yield": f"{int(round(chosen_yield))}%", "smiles": stored_smiles, "comment": None})
            
        step_max = max(step_yields)
        if step_max > current_global_max: current_global_max = step_max
        max_yield_history.append(current_global_max)
        print(f"[{model_name.split('/')[-1]} | Campaign {campaign_id}] Step {iteration}/14 complete. Max yield: {current_global_max:.1f}%")
            
    return max_yield_history

def run_prospective_experiment(models, num_campaigns=5):
    """Launches the full multithreaded AI discovery campaign suite."""
    print(f"\n==================== STARTING PARALLEL ACTIVE LEARNING ====================")
    results_dict = {m: [] for m in models}
    
    for target_model in models:
        print(f"\nDispatching {num_campaigns} AI Campaigns for {target_model.split('/')[-1]}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_campaigns) as executor:
            future_to_task = {executor.submit(run_prospective_campaign, i, target_model): i for i in range(1, num_campaigns + 1)}
            for future in concurrent.futures.as_completed(future_to_task):
                c_id = future_to_task[future]
                try:
                    res = future.result()
                    if res: results_dict[target_model].append(res)
                except Exception as e:
                    print(f">> [Failure] Campaign {c_id} crashed: {e}")
                    
    return results_dict