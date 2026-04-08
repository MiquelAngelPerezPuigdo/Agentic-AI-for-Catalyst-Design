import json
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