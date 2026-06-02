PROMPT_TEMPLATE = """
Consider the following chemical reaction: {reaction_name}.
{context_block}

Based on chemical principles (and the provided context if any), answer:
1. What is the mechanism of this reaction? Describe the elementary steps.
2. What properties of catalysts will be favorable according to this mechanism?
"""

EVAL_PROMPT_TEMPLATE = """
You are an expert computational chemist. Rate the response on a scale of 0 to 10.

Reaction: {reaction_name}
Context Provided to Model: {has_context}

RUBRIC:
0-2: Hallucinated or fundamentally wrong chemistry.
3-4: Correct reaction name, but incorrect or vague mechanism steps.
5-6: Mostly correct mechanism, but misses key intermediates or stereochemical nuances.
7-8: High accuracy, matches the provided paper context well.
9-10: Exceptional detail, correctly identifies catalyst electronics/sterics and all transition states.

GROUND TRUTH (Gold Standard): 
{ground_truth}

MODEL RESPONSE: 
{response_text}

Return ONLY valid JSON:
{{
    "score": <int>,
    "justification": "<string>"
}}
"""

STRICT_INSTRUCTIONS_BASIC = """
CRITICAL INSTRUCTIONS:
1. ANTI-VERBOSITY: Write exactly 1 concise paragraph (max 100 words) summarizing your logic.
2. ANTI-LAZINESS: You MUST score EVERY SINGLE catalyst provided (exactly {num_catalysts} items).
3. FORMAT: Do NOT use a vertical list. Output all scores on a SINGLE CONTINUOUS LINE under the exact heading [SCORES], separated by commas. You must end the line with the exact word [END_OF_SCORING].
"""

STRICT_INSTRUCTIONS_ADVANCED = """
CRITICAL INSTRUCTIONS:
1. STRUCTURED REASONING: Before scoring, provide your chemical logic strictly in this format:
   - MECHANISM RULE (Max 50 words): What specific sterics/electronics are ideal?
   - LIGAND CLASSIFICATION (Max 150 words): Group the provided SMILES into Good/Average/Poor.
2. ANTI-LAZINESS: You MUST score EVERY SINGLE catalyst provided (exactly {num_catalysts} items).
3. FORMAT: Do NOT use a vertical list. Output all scores on a SINGLE CONTINUOUS LINE under the exact heading [SCORES], separated by commas. You must end the line with the exact word [END_OF_SCORING].
"""

PROMPTS_TEXT_MODE = {
    "Level 1 (Zero-Shot)": """Reaction: {reaction_definition}.\nCatalysts:\n{catalyst_list}\nScore these catalysts from 0 to 100.""" + STRICT_INSTRUCTIONS_BASIC,
    "Level 2 (CoT: Auto-Mechanism)": """Reaction: {reaction_definition}.\nStep 1: State the likely mechanism.\nStep 2: Score the catalysts:\n{catalyst_list}""" + STRICT_INSTRUCTIONS_BASIC,
    "Level 3 (CoT: Mechanism -> Properties)": """Reaction: {reaction_definition}.\nStep 1: State the likely mechanism and required properties.\nStep 2: Score the catalysts:\n{catalyst_list}""" + STRICT_INSTRUCTIONS_BASIC,
    "Level 4 (Supplied Mechanism -> Properties)": """Reaction: {reaction_definition}.\nMechanism: {mechanism_text}\nScore the following catalysts:\n{catalyst_list}""" + STRICT_INSTRUCTIONS_ADVANCED
}

SYSTEM_PROMPT_JSON = """You are an expert computational chemist and a rigid JSON data-formatting engine.

CRITICAL INSTRUCTIONS (FAILURE WILL CRASH THE PIPELINE):
1. You MUST evaluate exactly {num_catalysts} catalysts. Missing even one is a fatal error.
2. NEVER use ellipses ("..."), "and so on", or skip any items.
3. You MUST output your response as a strictly valid JSON object. Do not include any text outside the JSON block.

REQUIRED JSON STRUCTURE:
{{
  "reasoning": "Briefly state your mechanism rule and ligand classification here.",
  "scores": {{
    "Catalyst_ID_1": 85.5,
    "Catalyst_ID_2": 42.0
  }}
}}
"""

PROMPTS_JSON_MODE = {
    "Level 5 (Paper Context Zero-Shot)": """--- BACKGROUND LITERATURE CONTEXT ---\n{related_paper_text}\n--- END BACKGROUND CONTEXT ---\nReaction: {reaction_definition}\nTASK:\n1. Formulate a hypothesis.\n2. Score EXACTLY {num_catalysts} catalysts.\nCATALYSTS TO SCORE:\n{catalyst_list}\nReturn ONLY a valid JSON object.""",
    "Level 6 (Paper Context -> Mechanism -> Properties)": """--- ACTUAL PUBLISHED PAPER CONTEXT ---\n{actual_paper_text}\n--- END PUBLISHED PAPER CONTEXT ---\nReaction: {reaction_definition}\nTASK:\n1. Formulate a hypothesis.\n2. Score EXACTLY {num_catalysts} catalysts.\nCATALYSTS TO SCORE:\n{catalyst_list}\nReturn ONLY a valid JSON object."""
}

ALL_LEVEL_KEYS = list(PROMPTS_TEXT_MODE.keys()) + list(PROMPTS_JSON_MODE.keys())

