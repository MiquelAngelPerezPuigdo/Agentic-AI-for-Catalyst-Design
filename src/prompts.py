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