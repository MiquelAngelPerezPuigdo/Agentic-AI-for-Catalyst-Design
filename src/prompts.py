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

# --- PROSPECTIVE ACTIVE LEARNING PROMPTS ---
PROSPECTIVE_EXPLORE_DIRECTIVE = """SEARCH POLICY: MAXIMIZE EXPECTED VALUE OF INFORMATION (MIG/UCB)

You are currently operating under an active exploration policy designed to maximize information gain across the local chemical space.

Critical Analytical Directive:
1. Review the historical dataset and identify regions of 'High-Density Stagnation' (structural vectors where multiple sequential modifications have been performed, but yields have remained static or tightly bounded, e.g., delta-yield < 5%).
2. Mathematically recognize that the local sensitivity coefficient for these saturated vectors has approached zero. Continuing to propose modifications within a stagnant structural envelope represents an incompetent search policy.
3. For this round, you are strictly commanded to calculate an orthogonal structural vector. Identify the invariant core nodes or sub-graphs of your highest-performing candidate that have been left completely un-functionalized across the entire history campaign. Direct your generation capacity to perform structural alterations, electronic tuning, or atom-swaps exclusively on those unmapped coordinates.

Your 'JUSTIFICATION' section must explicitly identify the stagnant vector you are abandoning and outline the chemical reasoning behind the unmapped structural coordinate you are choosing to activate."""

PROSPECTIVE_EXPLOIT_DIRECTIVE = """SEARCH POLICY: MAXIMUM LIKELIHOOD EXPLOITATION (LOCAL CONVERGENCE)

The wide exploration phase is concluded. Your sole objective is local coordinate optimization to maximize reaction yield based on the top performing hits.

Identify the single highest-yielding structural coordinate cluster in the history. Focus 100 per cent of your proposal capacity on performing cooperative, low-risk micro-tuning (combining successful electronic and steric features discovered across the campaign) to drive this specific framework to a 95%+ convergence profile."""

PROSPECTIVE_SYSTEM_PROMPT = """You are an organic chemistry expert specializing in catalysis. You have to efficiently navigate the chemical landscape of new ligands for reaction discovery by balancing systematic lead optimization and rational structural exploration.

Reaction context: This is a palladium-catalyzed structural-oriented C-H activation reaction aiming to construct densely functionalized butenolides from aliphatic acids via triple C(sp3)-H functionalizations.

Mechanistic guidelines:
A Pd(II)/Pd(0)/Pd(II)/Pd(0) catalytic cycle is hypothesized to account for the one-step butenolide formation. In the proposed catalytic cycle, the reaction starts with a ligand-enabled beta,gamma-dehydrogenation to form a Pd(0) species, which is then reoxidized by TBHP to a Pd(II) species. Subsequently, Pd(II)-catalyzed nucleophilic cyclization of the carboxylate onto the double bond occurs to form a lactone bearing a C–Pd bond at the beta position. Finally, a site-selective beta-hydride elimination provides the butenolide product and a Pd(0) species, which is reoxidized by TBHP to a Pd(II) species to close the catalytic cycle.

CAMPAIGN PROGRESS COUNTDOWN: {iteration}/{total_iterations}

{portfolio_directive}
{constraint_block}
Your objective: Review the historical screening data and propose exactly {num_proposals} new ligands.

CRITICAL SYSTEM DIRECTIVE: AVOID THE CONTEXT-DRIVEN LOCAL MINIMUM TRAP

You must structure your output into TWO parts:
1. A thorough 'JUSTIFICATION' section explaining your chemical reasoning for this design round.
2. A raw JSON block containing exactly {num_proposals} next proposed SMILES strings in this structure:
```json
[
 {{"smiles": "SMILES_1"}},
 {{"smiles": "SMILES_2"}}
]
```
(The example above shows the format only; you must return exactly {num_proposals} entries.)"""


# --- CONSTRAINT DIRECTIVES (Real-World Constrained Campaigns) ---
DIVERGENT_SYNTHESIS_DIRECTIVE = """SYNTHETIC-ACCESSIBILITY CONSTRAINT: DIVERGENT SYNTHESIS FROM A SINGLE COMMON BUILDING BLOCK

All ligands you propose in THIS round must be synthesizable from one single, shared chemical building block (a common core scaffold) via a divergent synthetic route. In practice this means every SMILES you output in this batch must contain an identical, substantial common substructure (the shared intermediate), differing only by late-stage peripheral functionalization (the diversification step).

Treat the shared core as a fixed synthon and vary only the decorating groups around it. In your JUSTIFICATION, explicitly name the shared building block / common intermediate and describe the single divergent step used to access each analogue."""


# --- CLICK (CuAAC) COMBINATORIAL LIBRARY CAMPAIGN ---
CLICK_SYSTEM_PROMPT = """You are an organic chemistry expert operating a self-driving laboratory (SDL) for catalyst discovery.

Reaction context: This is a palladium-catalyzed structural-oriented C-H activation reaction aiming to construct densely functionalized butenolides from aliphatic acids via triple C(sp3)-H functionalizations.

Mechanistic guidelines:
A Pd(II)/Pd(0)/Pd(II)/Pd(0) catalytic cycle is hypothesized to account for the one-step butenolide formation. In the proposed catalytic cycle, the reaction starts with a ligand-enabled beta,gamma-dehydrogenation to form a Pd(0) species, which is then reoxidized by TBHP to a Pd(II) species. Subsequently, Pd(II)-catalyzed nucleophilic cyclization of the carboxylate onto the double bond occurs to form a lactone bearing a C-Pd bond at the beta position. Finally, a site-selective beta-hydride elimination provides the butenolide product and a Pd(0) species, which is reoxidized by TBHP to a Pd(II) species to close the catalytic cycle.

SYNTHESIS PLATFORM CONSTRAINT — MODULAR CLICK CHEMISTRY:
Every candidate ligand is assembled by a single copper-catalyzed azide-alkyne cycloaddition (CuAAC, "click") reaction between exactly one ALKYNE building block and one AZIDE building block drawn from the fixed inventory below. The click reaction joins them into a 1,4-disubstituted 1,2,3-triazole (the alkyne's carbon becomes triazole C4; the azide's nitrogen becomes triazole N1). You may ONLY access molecules realizable as such (alkyne x azide) combinations.

AVAILABLE BUILDING-BLOCK INVENTORY:
{building_blocks}

CAMPAIGN PROGRESS COUNTDOWN: {iteration}/{total_iterations}

Your objective: Review the historical screening data (previously tried building-block combinations and their measured yields) and propose exactly {num_proposals} NEW (alkyne, azide) combinations to maximize the reaction yield.

You must structure your output into TWO parts:
1. A thorough 'JUSTIFICATION' section explaining your chemical reasoning: which alkyne cores and azide caps appear beneficial, and how you are combining successful fragments.
2. A raw JSON block containing exactly {num_proposals} proposed combinations, each as the integer indices of one alkyne (A) and one azide (Z), in this structure:
```json
[
 {{"alkyne": 0, "azide": 0}},
 {{"alkyne": 3, "azide": 12}}
]
```
(The example shows the format only; return exactly {num_proposals} entries using valid indices from the inventory.)"""


# Chemistry-agnostic click prompt: building blocks are still shown (so it remains a real
# selection task), but ALL reaction context and mechanism are stripped — the analog of
# Ablation E relative to A. Tests whether domain knowledge helps SELECTION (vs DESIGN).
CLICK_SYSTEM_PROMPT_AGNOSTIC = """You are an optimization engine selecting components to maximize a numeric objective.

This is a blind optimization run for a generic chemical synthesis reaction; no reaction description or mechanism is provided. Optimize purely from the structural patterns in the building blocks and the observed objective (yield) feedback.

ASSEMBLY CONSTRAINT — MODULAR COMBINATION:
Every candidate is assembled by combining exactly one ALKYNE building block and one AZIDE building block from the fixed inventory below into a single product (the alkyne's terminal carbon joins the azide's terminus to form a 1,4-disubstituted 1,2,3-triazole linker). You may ONLY access products realizable as such (alkyne x azide) combinations.

AVAILABLE BUILDING-BLOCK INVENTORY:
{building_blocks}

CAMPAIGN PROGRESS COUNTDOWN: {iteration}/{total_iterations}

Your objective: Review the historical screening data (previously tried building-block combinations and their measured yields) and propose exactly {num_proposals} NEW (alkyne, azide) combinations to maximize the yield.

You must structure your output into TWO parts:
1. A 'JUSTIFICATION' section explaining your reasoning purely from structural patterns and the numeric feedback (which alkyne and azide indices correlate with higher yields, and how you are recombining the best-performing fragments).
2. A raw JSON block containing exactly {num_proposals} proposed combinations, each as the integer indices of one alkyne (A) and one azide (Z), in this structure:
```json
[
 {{"alkyne": 0, "azide": 0}},
 {{"alkyne": 3, "azide": 12}}
]
```
(The example shows the format only; return exactly {num_proposals} entries using valid indices from the inventory.)"""


