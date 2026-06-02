import json
import re
from src.api import call_openrouter
from src.prompts import EVAL_PROMPT_TEMPLATE
from config import JUDGE_MODEL


def evaluate_response(rxn_name, ground_truth, response_text, has_context):
    """Sends the model's response to the Judge LLM for scoring."""
    eval_prompt = EVAL_PROMPT_TEMPLATE.format(
        reaction_name=rxn_name,
        has_context="Yes" if has_context else "No",
        ground_truth=ground_truth,
        response_text=response_text
    )
    res = call_openrouter(JUDGE_MODEL, eval_prompt, temperature=0)
    
    try:
        # Extract just the JSON block in case the judge adds conversational text
        clean_json = res[res.find("{"):res.rfind("}")+1]
        data = json.loads(clean_json)
        return data.get('score', 0), data.get('justification', "No justification provided.")
    except Exception as e: 
        return 0, f"Evaluation failed to parse JSON. Raw response: {res}"
    
def extract_scores_text(response_text):
    scores = {}
    parts = response_text.split("[SCORES]")
    text_to_parse = parts[-1] if len(parts) > 1 else response_text
    pairs = re.findall(r'(Catalyst_[a-zA-Z0-9_]+)\s*(?:\||:|-)\s*([0-9]*\.?[0-9]+)', text_to_parse)
    for ligand, score in pairs:
        try: scores[ligand] = float(score)
        except ValueError: pass
    return scores

def extract_scores_json(response_text):
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]

        data = json.loads(json_str)
        scores = data.get("scores", {})
        
        clean_scores = {}
        for k, v in scores.items():
            try: clean_scores[k] = float(v)
            except ValueError: pass
        return clean_scores
    except Exception as e:
        print(f"\n    [!] JSON Parsing Failed. Model output invalid JSON.")
        return {}

def calculate_top_5_overlap(predicted_list, true_list):
    if not predicted_list: return 0 
    valid_predicted = [p for p in predicted_list if p in true_list]
    predicted_top_k = set(valid_predicted[:5]) if len(valid_predicted) >= 5 else set(valid_predicted)
    return len(set(true_list[:5]).intersection(predicted_top_k))