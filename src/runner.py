import concurrent.futures
import pandas as pd
from tqdm import tqdm
from src.api import call_openrouter
from src.evaluation import evaluate_response
from src.prompts import PROMPT_TEMPLATE
from config import ALL_MODELS, FRONTIER_MODELS, ITERATIONS, JUDGE_MODEL
try:
    from config import USE_PROMPT_CACHING
except ImportError:
    USE_PROMPT_CACHING = False
# --- ADDED FOR BENCHMARK 2: LIGAND RANKING ---
from src.ligand_data import RANKING_TASKS
from src.prompts import PROMPTS_TEXT_MODE, PROMPTS_JSON_MODE, SYSTEM_PROMPT_JSON, ALL_LEVEL_KEYS
from src.evaluation import extract_scores_text, extract_scores_json, calculate_top_5_overlap
import json
# --- ADDED FOR PROSPECTIVE CASE ---
import re
from src.ligand_data import INITIAL_SEED
from src.paths import out_path

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
    
    # A "group" is one (model, reaction, context) combination whose ITERATIONS calls
    # share a byte-identical prompt (including the up-to-50k-char paper context).
    groups = []
    for model in ALL_MODELS:
        for rxn_key, rxn_data in reactions_dict.items():
            for use_context in [True, False]:
                groups.append((model, rxn_key, rxn_data, use_context))

    if USE_PROMPT_CACHING and ITERATIONS > 1:
        # Prime-then-fan-out: run iteration 0 of every group first (in parallel across
        # groups) to warm each provider-side cache, THEN submit the remaining identical
        # iterations so they read the cached prefix instead of re-paying for prefill.
        print("-> Prompt caching ON: priming shared prompt prefixes before fan-out...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Phase 2a: prime one call per group.
            prime_futures = {
                executor.submit(process_single_task, model, rxn_key, rxn_data, use_context, 0): (model, rxn_key, rxn_data, use_context)
                for (model, rxn_key, rxn_data, use_context) in groups
            }
            for future in tqdm(concurrent.futures.as_completed(prime_futures), total=len(prime_futures), desc="Priming caches"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"\n[!] A priming task generated an exception: {exc}")

            # Phase 2b: fan out the remaining ITERATIONS-1 calls per group (cache hits).
            fanout_tasks = [
                (model, rxn_key, rxn_data, use_context, i)
                for (model, rxn_key, rxn_data, use_context) in groups
                for i in range(1, ITERATIONS)
            ]
            fanout_futures = {
                executor.submit(process_single_task, *task_args): task_args
                for task_args in fanout_tasks
            }
            for future in tqdm(concurrent.futures.as_completed(fanout_futures), total=len(fanout_futures), desc="API Calls (cached)"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"\n[!] A task generated an exception: {exc}")
    else:
        tasks = [
            (model, rxn_key, rxn_data, use_context, i)
            for (model, rxn_key, rxn_data, use_context) in groups
            for i in range(ITERATIONS)
        ]
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

        # Use a progress bar for the total expected successes
        total_required_successes = len(trackers) * required_iterations
        pbar = tqdm(total=total_required_successes, desc="Ligand Successes")

        # 0. (Optional) Prime each combination's shared prompt prefix with ONE serial-per-group
        #    call before fan-out, so the remaining identical iterations hit a warm cache.
        #    Levels 5 & 6 embed full paper text, so this prefix is large and worth caching.
        if USE_PROMPT_CACHING and required_iterations > 1:
            print("-> Prompt caching ON: priming shared prompt prefixes before fan-out...")
            prime_futures = {}
            for (model, task_key, level_name), tracker in trackers.items():
                task_data = RANKING_TASKS[task_key]
                future = executor.submit(process_ligand_task, model, task_key, level_name, 0, task_data)
                prime_futures[future] = (model, task_key, level_name, 0, task_data)
                tracker["attempts_submitted"] += 1
            for future in tqdm(concurrent.futures.as_completed(prime_futures), total=len(prime_futures), desc="Priming caches"):
                model, task_key, level_name, _, task_data = prime_futures[future]
                tracker = trackers[(model, task_key, level_name)]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"\n[!] Exception during priming for {model.split('/')[-1]} on {task_key}/{level_name}: {e}")
                    res = None
                if res:
                    res["Iteration"] = tracker["successes_got"]
                    results.append(res)
                    tracker["successes_got"] += 1
                    pbar.update(1)

        # 1. Submit initial tasks. With priming, one attempt per group already ran, so we
        #    fan out the remaining (required_iterations - primed) attempts; otherwise submit
        #    the full required_iterations as before.
        primed = USE_PROMPT_CACHING and required_iterations > 1
        for (model, task_key, level_name), tracker in trackers.items():
            task_data = RANKING_TASKS[task_key]
            initial_to_submit = (required_iterations - 1) if primed else required_iterations
            start_idx = tracker["attempts_submitted"]
            for offset in range(initial_to_submit):
                i = start_idx + offset
                future = executor.submit(process_ligand_task, model, task_key, level_name, i, task_data)
                future_to_info[future] = (model, task_key, level_name, i, task_data)
                tracker["attempts_submitted"] += 1
                tracker["pending_attempts"] += 1
        
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

# --- Divergent-synthesis PASS/FAIL thresholds (literature-grounded) ---
# Murcko = 1.0: all molecules MUST share the identical generic ring scaffold (primary, non-negotiable gate).
# MCS >= 0.5: shared core is at least half of each molecule's heavy atoms (standard "common scaffold" convention).
# Tanimoto >= 0.4: classic Patterson/Maggiora "same chemical series" similarity threshold.
DIVERGENT_MURCKO_MIN = 1.0
DIVERGENT_MCS_MIN = 0.5
DIVERGENT_TANIMOTO_MIN = 0.4

def compute_scaffold_conservation(smiles_list):
    """Audit how well a batch of ligands derives from a single common building block (divergent synthesis).

    Reports THREE complementary metrics plus a binary PASS verdict:
      1. mcs_conservation_score: mean fraction of each molecule's heavy atoms belonging to the batch's
         Maximum Common Substructure (MCS) core. Continuous measure of "how much" core is shared.
      2. murcko_scaffold_agreement: fraction of molecules sharing the single most common Bemis-Murcko
         generic scaffold. Cleanest binary-style "same core ring system" definition.
      3. mean_pairwise_tanimoto: average Morgan-fingerprint Tanimoto similarity across all batch pairs.
         Overall chemical-similarity sanity check.

    'divergent_pass' is True when the primary Murcko gate is met AND at least one corroborating
    metric (MCS or Tanimoto) clears its threshold.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS, DataStructs
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        return None

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    n = len(mols)
    base = {
        "num_molecules": n,
        "shared_core_smarts": None,
        "core_num_atoms": 0,
        "mcs_conservation_score": 0.0,
        "murcko_scaffold_agreement": 0.0,
        "most_common_murcko": None,
        "mean_pairwise_tanimoto": 0.0,
    }
    if n < 2:
        return base

    # --- 1. MCS conservation ---
    res = rdFMCS.FindMCS(
        mols,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    if not res.canceled and res.numAtoms > 0:
        base["shared_core_smarts"] = res.smartsString
        base["core_num_atoms"] = res.numAtoms
        fr = [res.numAtoms / m.GetNumHeavyAtoms() for m in mols if m.GetNumHeavyAtoms() > 0]
        base["mcs_conservation_score"] = round(sum(fr) / len(fr), 4) if fr else 0.0

    # --- 2. Bemis-Murcko generic scaffold agreement ---
    scaffolds = []
    for m in mols:
        try:
            murcko = MurckoScaffold.GetScaffoldForMol(m)
            generic = MurckoScaffold.MakeScaffoldGeneric(murcko)
            scaffolds.append(Chem.MolToSmiles(generic))
        except Exception:
            scaffolds.append(None)
    valid = [s for s in scaffolds if s]
    if valid:
        from collections import Counter
        most_common, count = Counter(valid).most_common(1)[0]
        base["most_common_murcko"] = most_common
        base["murcko_scaffold_agreement"] = round(count / n, 4)

    # --- 3. Mean pairwise Tanimoto (Morgan r=2) ---
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [gen.GetFingerprint(m) for m in mols]
        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        base["mean_pairwise_tanimoto"] = round(sum(sims) / len(sims), 4) if sims else 0.0
    except Exception:
        pass

    # --- Binary PASS verdict (primary Murcko gate + at least one corroborating metric) ---
    primary = base["murcko_scaffold_agreement"] >= DIVERGENT_MURCKO_MIN
    corroborating = (base["mcs_conservation_score"] >= DIVERGENT_MCS_MIN) or \
                    (base["mean_pairwise_tanimoto"] >= DIVERGENT_TANIMOTO_MIN)
    base["divergent_pass"] = bool(primary and corroborating)

    return base

def run_prospective_campaign(campaign_id, model_name, surrogate, fp_gen, dataset_dict, ablation_mode="A", total_iterations=14, save_details=False, num_proposals=3, constraint=None):
    # Only import heavy dependencies if this function is actually called
    try:
        from rdkit import Chem
        from src.surrogate import score_ligand
    except ImportError:
        return [] # Returns empty if RDkit isn't installed

    from src.prompts import PROSPECTIVE_EXPLORE_DIRECTIVE, PROSPECTIVE_EXPLOIT_DIRECTIVE, PROSPECTIVE_SYSTEM_PROMPT, DIVERGENT_SYNTHESIS_DIRECTIVE
    
    print(f"[{model_name.split('/')[-1]} | Campaign {campaign_id} | Ablation {ablation_mode}] Starting with {total_iterations} steps ({num_proposals}/step)...")
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
                history_records.append({"yield": yield_val, "smiles": smiles_to_store, "raw_smiles": smiles_val, "comment": None})

    TOTAL_ITERATIONS = total_iterations
    max_yield_history = []
    current_global_max = 0.0
    campaign_logs = []

    def get_yield_num(record):
        try:
            return float(record["yield"].replace("%", "").strip())
        except Exception:
            return 0.0

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        # ===== FINAL UNBIASED ABLATION SCHEMA (A-F) =====
        # A = Baseline: Shuffled SMILES + SAR ladder sort + Full chemistry (no portfolio)
        # B = A - SAR ladder (insertion order)
        # C = A - SMILES Shuffle (Canonical instead of Shuffled)
        # D = A - Mechanism (reaction context kept)
        # E = A - Chemistry (full chemical blackout)
        # F = Full ablation (Canonical SMILES + insertion order + chemical blackout)

        # 1. Manage SAR Ladder Sorting (kept for all except B and F, which use raw insertion order)
        current_history = list(history_records)
        if ablation_mode in ["B", "F"]:
            pass  # Keep the chronological order in which results arrived (no SAR ladder)
        else:
            current_history.sort(key=get_yield_num, reverse=True)

        # 2. Manage SMILES representation (canonical for C and F; shuffled/doRandom for A, B, D, E)
        shuffled_history_lines = []
        for record in current_history:
            try:
                mol_obj = Chem.MolFromSmiles(record["smiles"])
                if mol_obj:
                    if ablation_mode in ["C", "F"]:
                        enumerated_smiles = Chem.MolToSmiles(mol_obj, canonical=True)
                    else:
                        enumerated_smiles = Chem.MolToSmiles(mol_obj, doRandom=True)
                else:
                    enumerated_smiles = record.get("raw_smiles", record["smiles"])
            except Exception: 
                enumerated_smiles = record.get("raw_smiles", record["smiles"])
            shuffled_history_lines.append(f"{record['yield']}\t{enumerated_smiles}")
            
        history_text_for_llm = "\n".join(shuffled_history_lines)

        # 3. Manage Portfolio Directive (all modes now use the standard, unbiased optimization prompt)
        directive = "SEARCH POLICY: STANDARD SYSTEMATIC OPTIMIZATION\nPropose new ligands to improve the yield based on the provided history."

        # Inject any real-world constraint directive (e.g. divergent synthesis).
        # The divergent constraint is RELAXED once the campaign reaches a strong lead (>=80%),
        # so the agent is free to make the final non-divergent move needed to reach the 89% optimum.
        DIVERGENT_RELEASE_YIELD = 80.0
        divergent_active = (constraint == "divergent") and (current_global_max < DIVERGENT_RELEASE_YIELD)
        if divergent_active:
            constraint_block = "\n" + DIVERGENT_SYNTHESIS_DIRECTIVE + "\n"
        else:
            constraint_block = ""

        sys_prompt = PROSPECTIVE_SYSTEM_PROMPT.format(iteration=iteration, total_iterations=TOTAL_ITERATIONS, portfolio_directive=directive, constraint_block=constraint_block, num_proposals=num_proposals)

        # 4. Manage Mechanism / Chemical stripping
        if ablation_mode == "D":
            # Strip mechanism but keep reaction context
            guidelines_str = "Mechanistic guidelines:\nA Pd(II)/Pd(0)/Pd(II)/Pd(0) catalytic cycle is hypothesized to account for the one-step butenolide formation. In the proposed catalytic cycle, the reaction starts with a ligand-enabled beta,gamma-dehydrogenation to form a Pd(0) species, which is then reoxidized by TBHP to a Pd(II) species. Subsequently, Pd(II)-catalyzed nucleophilic cyclization of the carboxylate onto the double bond occurs to form a lactone bearing a C–Pd bond at the beta position. Finally, a site-selective beta-hydride elimination provides the butenolide product and a Pd(0) species, which is reoxidized by TBHP to a Pd(II) species to close the catalytic cycle."
            sys_prompt = sys_prompt.replace(guidelines_str, "Mechanistic guidelines:\nNone provided. Optimize based on general ligand design principles.")
        elif ablation_mode in ["E", "F"]:
            # Chemical Blackout: Strip both reaction context and mechanism completely!
            guidelines_str = "Mechanistic guidelines:\nA Pd(II)/Pd(0)/Pd(II)/Pd(0) catalytic cycle is hypothesized to account for the one-step butenolide formation. In the proposed catalytic cycle, the reaction starts with a ligand-enabled beta,gamma-dehydrogenation to form a Pd(0) species, which is then reoxidized by TBHP to a Pd(II) species. Subsequently, Pd(II)-catalyzed nucleophilic cyclization of the carboxylate onto the double bond occurs to form a lactone bearing a C–Pd bond at the beta position. Finally, a site-selective beta-hydride elimination provides the butenolide product and a Pd(0) species, which is reoxidized by TBHP to a Pd(II) species to close the catalytic cycle."
            context_str = "Reaction context: This is a palladium-catalyzed structural-oriented C-H activation reaction aiming to construct densely functionalized butenolides from aliphatic acids via triple C(sp3)-H functionalizations."
            
            sys_prompt = sys_prompt.replace(context_str, "Reaction context: This is an optimization run for a generic chemical synthesis reaction.")
            sys_prompt = sys_prompt.replace(guidelines_str, "Mechanistic guidelines:\nNone provided. This is a blind search to optimize the yield of the target product based exclusively on structural patterns in the history.")

        user_prompt = f"Here is the history of tried ligands and their yields:\n\n{history_text_for_llm}\n\nThink carefully, output your justification, and provide your {num_proposals} next proposed ligands."
            
        max_gen_attempts = 5
        raw_response = "API_ERROR_OR_UNPARSABLE"
        for attempt in range(max_gen_attempts):
            try:
                # Reuses clean API block!
                response = call_openrouter(model_name, user_prompt, system_prompt=sys_prompt) 
                raw_response = response
                json_str = clean_json_markdown(response)
                proposed_ligands = json.loads(json_str.strip())
                smiles_list = [item['smiles'] for item in proposed_ligands]
                
                if len(smiles_list) != num_proposals: raise ValueError(f"Did not generate exactly {num_proposals}.")
                
                proposed_canonicals = []
                existing_canonicals = set([r["smiles"] for r in history_records])
                for s in smiles_list:
                    m = Chem.MolFromSmiles(s, sanitize=False)
                    if m is None: raise ValueError("Invalid SMILES")
                    Chem.SanitizeMol(m)
                    can_s = Chem.MolToSmiles(m, canonical=True)
                    
                    # 5. No duplicates checks (Ablation C was removed entirely, duplicates are never allowed now)
                    if can_s in existing_canonicals or can_s in proposed_canonicals: 
                        raise ValueError("Duplicate SMILES")
                    proposed_canonicals.append(can_s)

                # 6. ENFORCE divergent-synthesis constraint while it is active:
                #    the whole batch must pass the shared-scaffold audit, else reject & retry.
                if divergent_active:
                    audit = compute_scaffold_conservation(proposed_canonicals)
                    if not (audit and audit.get("divergent_pass")):
                        raise ValueError("Batch failed divergent-synthesis constraint")

                break # Success! Break out of retry loop.
            except Exception:
                if attempt == max_gen_attempts - 1:
                    import random
                    existing_canonicals = set([r["smiles"] for r in history_records])
                    untried_candidates = [smiles for smiles in dataset_dict if smiles not in existing_canonicals]
                    if divergent_active and len(untried_candidates) >= num_proposals:
                        # Constraint-aware fallback: pick an untried molecule, then gather its
                        # nearest scaffold-mates so the fallback batch still satisfies divergence.
                        anchor = random.choice(untried_candidates)
                        anchor_pool = sorted(
                            untried_candidates,
                            key=lambda s: -(compute_scaffold_conservation([anchor, s]) or {}).get("mcs_conservation_score", 0.0),
                        )
                        smiles_list = anchor_pool[:num_proposals]
                    elif len(untried_candidates) >= num_proposals:
                        smiles_list = random.sample(untried_candidates, num_proposals)
                    else:
                        hardcoded_defaults = ["C1=CC=C(C=C1)CN2C=C(C3=NC(=CC=C3)O)N=N2", "Oc1nc(C2=CN(CC3=CC=C(C=C3)C(F)(F)F)N=N2)c(C)cc1", "COC1=C(N=C(C=C1)C2=CN(CC3=CC=CC=C3)N=N2)O"]
                        smiles_list = [s for s in hardcoded_defaults if s not in existing_canonicals]
                        while len(smiles_list) < num_proposals:
                            smiles_list.append(f"CCCC{len(smiles_list)}C")

        # Evaluate and log
        step_yields = []
        scored_details = []
        for smiles in smiles_list:
            chosen_yield, stored_smiles = score_ligand(smiles, surrogate, fp_gen, dataset_dict, history_records)
            step_yields.append(chosen_yield)
            history_records.append({"yield": f"{int(round(chosen_yield))}%", "smiles": stored_smiles, "raw_smiles": smiles, "comment": None})
            scored_details.append({
                "proposed_smiles": smiles,
                "canonical_smiles": stored_smiles,
                "yield": chosen_yield
            })
            
        step_max = max(step_yields)
        if step_max > current_global_max: current_global_max = step_max
        max_yield_history.append(current_global_max)

        # Divergent-synthesis constraint audit: measure how well this batch shares a common scaffold (MCS).
        # Only steps where the constraint was ACTIVE (best < 80%) count toward the enforcement audit.
        scaffold_score = None
        if constraint == "divergent":
            scaffold_score = compute_scaffold_conservation([d["canonical_smiles"] for d in scored_details])
            if scaffold_score is not None:
                scaffold_score["constraint_active"] = bool(divergent_active)

        # Save step details if requested
        if save_details:
            step_log = {
                "step": iteration,
                "portfolio_directive": directive,
                "raw_llm_response": raw_response,
                "proposals": scored_details,
                "step_max_yield": step_max,
                "global_max_yield": current_global_max
            }
            if scaffold_score is not None:
                step_log["scaffold_conservation"] = scaffold_score
            campaign_logs.append(step_log)
        
        print(f"[{model_name.split('/')[-1]} | Campaign {campaign_id} | Ablation {ablation_mode}] Step {iteration}/{TOTAL_ITERATIONS} complete. Max yield: {current_global_max:.1f}%")
        
        # Early stopping if maximum possible database yield (89.0%) is achieved
        if current_global_max >= 89.0:
            print(f"[{model_name.split('/')[-1]} | Campaign {campaign_id} | Ablation {ablation_mode}] Target yield reached (89.0%+). Stopping early to save API cost!")
            # Pad the remaining steps in history with current max yield so plotting and averaging remain fully consistent
            remaining_steps = TOTAL_ITERATIONS - len(max_yield_history)
            if remaining_steps > 0:
                max_yield_history.extend([current_global_max] * remaining_steps)
            break
            
    # Save the detailed campaign logs if requested
    if save_details:
        log_file_path = out_path("campaign_details", f"campaign_details_{ablation_mode}_campaign_{campaign_id}.json")
        try:
            with open(log_file_path, "w") as f:
                json.dump(campaign_logs, f, indent=4)
            print(f"[+] Detailed campaign logs successfully saved to '{log_file_path}'.")
        except Exception as e:
            print(f"[!] Warning: Failed to save detailed campaign log: {e}")

    return max_yield_history


def run_click_campaign(campaign_id, model_name, surrogate, fp_gen, dataset_dict, total_iterations=14, save_details=False, num_proposals=3, chem_agnostic=False):
    """SDL click-library campaign: the LLM is fed only building blocks and proposes (alkyne, azide) index pairs.

    chem_agnostic=True strips all reaction context/mechanism from the prompt (analog of Ablation E),
    keeping only the building-block structures + numeric yield feedback. Probes whether chemical
    knowledge helps SELECTION (library navigation) the way it was found redundant for DE NOVO design.
    """
    try:
        from rdkit import Chem
        from src.surrogate import score_ligand
    except ImportError:
        return []

    from src.prompts import CLICK_SYSTEM_PROMPT, CLICK_SYSTEM_PROMPT_AGNOSTIC
    from src.click_library import get_library, format_building_blocks

    click_prompt = CLICK_SYSTEM_PROMPT_AGNOSTIC if chem_agnostic else CLICK_SYSTEM_PROMPT
    mode_tag = "Click-Agnostic" if chem_agnostic else "Click"

    lib = get_library()
    n_a, n_z = lib["n_alkynes"], lib["n_azides"]
    product_map = lib["product_map"]
    building_blocks = format_building_blocks()

    print(f"[{model_name.split('/')[-1]} | {mode_tag} Campaign {campaign_id}] Starting with {total_iterations} steps ({num_proposals}/step), library {n_a}x{n_z}={len(product_map)}...")

    history = []          # list of dicts: {alkyne, azide, smiles, yield}
    tried_pairs = set()
    max_yield_history = []
    current_global_max = 0.0
    campaign_logs = []

    for iteration in range(1, total_iterations + 1):
        # Build history text (sorted by yield, SAR-ladder style, matching Ablation A)
        hist_sorted = sorted(history, key=lambda r: r["yield"], reverse=True)
        if hist_sorted:
            hist_lines = [f"  (A{r['alkyne']}, Z{r['azide']}) -> {r['yield']:.1f}%" for r in hist_sorted]
            history_text = "\n".join(hist_lines)
        else:
            history_text = "  (no combinations tested yet)"

        sys_prompt = click_prompt.format(
            building_blocks=building_blocks, iteration=iteration,
            total_iterations=total_iterations, num_proposals=num_proposals,
        )
        user_prompt = (f"History of tested click combinations and their yields:\n\n{history_text}\n\n"
                       f"Propose your {num_proposals} next (alkyne, azide) index combinations to maximize yield.")

        pairs = None
        raw_response = "API_ERROR_OR_UNPARSABLE"
        for attempt in range(5):
            try:
                response = call_openrouter(model_name, user_prompt, system_prompt=sys_prompt)
                raw_response = response
                proposed = json.loads(clean_json_markdown(response).strip())
                cand = []
                for item in proposed:
                    ai, zi = int(item["alkyne"]), int(item["azide"])
                    if not (0 <= ai < n_a and 0 <= zi < n_z):
                        raise ValueError("Index out of range")
                    if (ai, zi) in tried_pairs or (ai, zi) in cand:
                        raise ValueError("Duplicate / already tried pair")
                    if (ai, zi) not in product_map:
                        raise ValueError("Pair does not yield a valid product")
                    cand.append((ai, zi))
                if len(cand) != num_proposals:
                    raise ValueError("Wrong number of proposals")
                pairs = cand
                break
            except Exception:
                if attempt == 4:
                    import random
                    avail = [p for p in product_map if p not in tried_pairs]
                    pairs = random.sample(avail, min(num_proposals, len(avail)))

        step_yields = []
        scored_details = []
        for (ai, zi) in pairs:
            smiles = product_map[(ai, zi)]
            chosen_yield, stored = score_ligand(smiles, surrogate, fp_gen, dataset_dict, [])
            tried_pairs.add((ai, zi))
            history.append({"alkyne": ai, "azide": zi, "smiles": stored, "yield": chosen_yield})
            step_yields.append(chosen_yield)
            scored_details.append({"alkyne": ai, "azide": zi, "product_smiles": smiles, "yield": chosen_yield})

        step_max = max(step_yields)
        if step_max > current_global_max:
            current_global_max = step_max
        max_yield_history.append(current_global_max)

        if save_details:
            campaign_logs.append({
                "step": iteration,
                "raw_llm_response": raw_response,
                "proposals": scored_details,
                "step_max_yield": step_max,
                "global_max_yield": current_global_max,
            })

        print(f"[{model_name.split('/')[-1]} | {mode_tag} Campaign {campaign_id}] Step {iteration}/{total_iterations} complete. Max yield: {current_global_max:.1f}%")

        if current_global_max >= 89.0:
            print(f"[{model_name.split('/')[-1]} | {mode_tag} Campaign {campaign_id}] Target yield reached (89.0%+). Stopping early.")
            remaining = total_iterations - len(max_yield_history)
            if remaining > 0:
                max_yield_history.extend([current_global_max] * remaining)
            break

    if save_details:
        label = "click_agnostic" if chem_agnostic else "click"
        path = out_path("campaign_details", f"campaign_details_{label}_campaign_{campaign_id}.json")
        try:
            with open(path, "w") as f:
                json.dump(campaign_logs, f, indent=4)
            print(f"[+] Detailed {mode_tag} campaign logs saved to '{path}'.")
        except Exception as e:
            print(f"[!] Warning: Failed to save {mode_tag} campaign log: {e}")

    return max_yield_history


def run_prospective_experiment(models, num_campaigns=5, ablation_mode="A", total_iterations=14, save_details=False, num_proposals=3, constraint=None):
    """Launches the full multithreaded AI discovery campaign suite."""
    print(f"\n==================== STARTING PARALLEL ACTIVE LEARNING (ABLATION: {ablation_mode}) ====================")
    results_dict = {m: [] for m in models}
    
    # Initialize the surrogate once per experiments run
    try:
        from src.surrogate import init_surrogate
        surrogate, fp_gen, dataset_dict = init_surrogate()
    except Exception as e:
        print(f"[!] Error initializing surrogate: {e}")
        return results_dict

    for target_model in models:
        print(f"\nDispatching {num_campaigns} AI Campaigns for {target_model.split('/')[-1]} (Ablation {ablation_mode})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_campaigns) as executor:
            future_to_task = {
                executor.submit(
                    run_prospective_campaign, 
                    i, 
                    target_model, 
                    surrogate, 
                    fp_gen, 
                    dataset_dict, 
                    ablation_mode=ablation_mode, 
                    total_iterations=total_iterations, 
                    save_details=save_details,
                    num_proposals=num_proposals,
                    constraint=constraint
                ): i for i in range(1, num_campaigns + 1)
            }
            for future in concurrent.futures.as_completed(future_to_task):
                c_id = future_to_task[future]
                try:
                    res = future.result()
                    if res: results_dict[target_model].append(res)
                except Exception as e:
                    print(f">> [Failure] Campaign {c_id} crashed: {e}")
                    
    return results_dict