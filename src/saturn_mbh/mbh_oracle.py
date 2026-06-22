"""
MBH catalyst-scoring oracle for Saturn (the essential MBH delta of this project's
Saturn fork, brought under the Agentic-AI-for-Catalyst-Design roof).

This module is a Saturn ``OracleComponent``: Saturn proposes candidate molecules,
this oracle filters them to valid tertiary-amine MBH catalysts (hard structural
guards to stop the RL agent reward-hacking) and batch-scores the survivors with an
LLM acting as a physical-organic-chemistry judge for the Morita-Baylis-Hillman
(MBH) reaction. DABCO is fixed at 50/100 as the calibration anchor.

It is imported *inside* a Saturn checkout at launch time (see ``launcher.py``),
so it depends on Saturn's ``oracles.oracle_component`` / ``oracles.dataclass``,
which are only importable when Saturn is on ``sys.path``. The LLM call uses the
OpenRouter (OpenAI-compatible) backend to match this repository's ``config.py``.
"""

import os
import json
import time
import concurrent.futures

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Provided by the Saturn checkout at runtime (injected onto sys.path by the launcher).
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters


# --- Hard structural guards (anti reward-hacking) ---------------------------
# Required core: a neutral tertiary amine that is NOT an amide / sulfonamide N.
TERTIARY_AMINE_PATTERN = Chem.MolFromSmarts("[NX3;H0;+0;!$(NC=O);!$(NS=O)]")
# Ban list: any N-H amine or any cationic nitrogen anywhere in the molecule.
FORBIDDEN_NITROGENS = Chem.MolFromSmarts("[NH1,NH2,NH3,n+1,N+1]")

MAX_MOL_WT = 250.0       # DABCO is 112 Da; cap forces compact scaffolds.
MAX_ROTATABLE_BONDS = 3  # MBH transition state punishes floppy chains.


def _extract_json(content: str) -> str:
    """Pull the JSON object out of an LLM reply, tolerating markdown fences or
    surrounding prose (Anthropic without forced JSON mode may add either)."""
    if content is None:
        return "{}"
    text = content.strip()
    if text.startswith("```"):
        # Drop the opening fence (``` or ```json) and the trailing fence.
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


def is_valid_mbh_catalyst(mol, max_mol_wt: float = MAX_MOL_WT,
                          max_rotatable_bonds: int = MAX_ROTATABLE_BONDS,
                          require_neutral: bool = False) -> bool:
    """Hard physical/structural filters that a candidate must pass before scoring.

    The MW and rotatable-bond ceilings are configurable so campaigns can explore
    a larger chemical space (defaults match the original compact-scaffold guards).

    With ``require_neutral=True``, any molecule carrying a non-zero formal charge
    on ANY atom is rejected. Real MBH catalysts are neutral tertiary amines, so
    this stops the RL agent reward-hacking by appending an anionic carboxylate
    (or any charged group) to please the LLM judge.
    """
    if mol is None:
        return False
    if not mol.HasSubstructMatch(TERTIARY_AMINE_PATTERN):
        return False
    if mol.HasSubstructMatch(FORBIDDEN_NITROGENS):
        return False
    if require_neutral and any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if Descriptors.MolWt(mol) > max_mol_wt:
        return False
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > max_rotatable_bonds:
        return False
    return True


class MBHcatalystscore(OracleComponent):
    """LLM-as-oracle that batch-scores tertiary-amine MBH catalysts (0-100, returned as 0-1).

    Specific parameters (set in the Saturn config JSON under ``specific_parameters``):
        api_key:          OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``.
        model_name:       OpenRouter model id (default: ``google/gemini-3.5-flash``).
        num_calls:        Replicate LLM calls per batch, averaged (default: 3).
        batch_size:       Molecules scored per prompt (default: 16).
        rate_limit_delay: Seconds to sleep between batches (default: 1.0).
    """

    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        sp = self.parameters.specific_parameters
        self.api_key = (
            sp.get("api_key")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "MBH oracle needs an API key: set 'api_key' in specific_parameters "
                "or export ANTHROPIC_API_KEY / OPENROUTER_API_KEY."
            )

        # Pick the backend from the key shape: Anthropic keys ('sk-ant-...') hit
        # Anthropic's OpenAI-compatible endpoint directly; everything else goes
        # through OpenRouter to match this repository's config.py.
        self.is_anthropic = self.api_key.startswith("sk-ant-")
        base_url = (
            "https://api.anthropic.com/v1/"
            if self.is_anthropic
            else "https://openrouter.ai/api/v1"
        )

        # Import lazily so the module is importable even where openai isn't installed.
        from openai import OpenAI
        # max_retries=0: we drive retries ourselves so the per-call timeout is the
        # exact hard ceiling (no hidden SDK backoff stacking on top of it).
        self.client = OpenAI(base_url=base_url, api_key=self.api_key, max_retries=0)

        default_model = "claude-opus-4-8" if self.is_anthropic else "google/gemini-3.5-flash"
        self.model_name = sp.get("model_name", default_model)
        self.num_calls = int(sp.get("num_calls", 3))
        self.batch_size = int(sp.get("batch_size", 16))
        self.rate_limit_delay = float(sp.get("rate_limit_delay", 1.0))
        self.max_tokens = int(sp.get("max_tokens", 4096))
        # Structural filter ceilings (configurable to widen the search space).
        self.max_mol_wt = float(sp.get("max_mol_wt", MAX_MOL_WT))
        self.max_rotatable_bonds = int(sp.get("max_rotatable_bonds", MAX_ROTATABLE_BONDS))
        # Enforce overall neutrality (reject any formally charged atom, e.g. a
        # carboxylate) so the agent can't reward-hack with anionic groups.
        self.require_neutral = bool(sp.get("require_neutral", False))
        # Hard per-call timeout (seconds): if one call (one batch attempt) exceeds
        # this, it is aborted and the SAME batch is asked again (see
        # full_batch_attempts) rather than scored 0 -- a slow answer must not be
        # mistaken for a bad one.
        self.request_timeout = float(sp.get("request_timeout", 600.0))
        # Fallback ladder for a batch whose call times out / errors:
        #   1. Re-ask the FULL batch up to `full_batch_attempts` times.
        #   2. If all fail, score each molecule individually (one call per mol) --
        #      a single molecule is fast and reasoning-light, so a slow full batch
        #      is retried cheaply one-by-one instead of being thrown away.
        #   3. Only if a molecule's individual call also fails is it scored 0.0.
        self.full_batch_attempts = int(sp.get("full_batch_attempts", 2))
        # Reasoning controls for heavy "thinking" models (e.g. Kimi/o-series via
        # OpenRouter): keep them from over-thinking forever. Only sent when set.
        #   reasoning_effort:     "low" | "medium" | "high"
        #   reasoning_max_tokens: hard cap on internal reasoning tokens.
        self.reasoning_effort = sp.get("reasoning_effort")
        self.reasoning_max_tokens = sp.get("reasoning_max_tokens")
        # Prompt caching: the large static MBH-mechanism block is identical across
        # every batch and across the `num_calls` replicate calls, so it is an ideal
        # cache prefix. Enabled by default; controlled from the Saturn config JSON.
        self.use_prompt_caching = bool(sp.get("use_prompt_caching", True))

    # The static instruction block is constant for the entire campaign, so it is
    # built once and reused as the cacheable prompt prefix. Only the trailing
    # {index: SMILES} JSON map varies per batch (appended as a separate block).
    _STATIC_PROMPT_PREFIX = (
        "You are an expert physical organic chemist evaluating molecules for the discovery of new tertiary amines that can catalyze the Morita-Baylis-Hillman (MBH) reaction.\n"
        "Consider the Morita-Baylis-Hillman (MBH) reaction of methyl acrylate (MA) "
        "with p-nitrobenzaldehyde (pNBA) in methanol.\n\n"

        "### Reaction Context & Mechanism\n"
        "1. Mechanistic Steps:\n"
        "   - Step 1: Nucleophilic attack of the tertiary amine on MA to form a zwitterionic enolate.\n"
        "   - Step 2: Aldol-type C-C bond formation via addition of the enolate to pNBA, forming a zwitterionic alkoxide.\n"
        "   - Step 3: Solvent-mediated proton transfer and subsequent elimination of the amine catalyst to yield the final MBH adduct.\n\n"

        "2. Catalyst Requirements (Tertiary Amines):\n"
        "   - High Nucleophilicity: Essential to efficiently initiate the reaction (Step 1).\n"
        "   - Low Steric Hindrance: Critical to allow attack and to avoid severe steric clashes in the highly congested C-C bond-forming transition state. Compact or bicyclic amines (e.g., DABCO, quinuclidine derivatives) generally outperform bulky acyclic amines (e.g., triethylamine).\n"
        "   - Leaving Group Ability: Must readily eliminate in Step 3 to turn over the catalytic cycle.\n\n"

        "3. Solvent Effects & Rate-Determining Step (RDS) in Methanol:\n"
        "   - Methanol serves as a strong hydrogen-bond donor, significantly stabilizing the highly polar zwitterionic intermediates.\n"
        "   - Because methanol powerfully facilitates proton transfer (Step 3), the Aldol addition (C-C bond formation in Step 2) becomes the strict Rate-Determining Step (RDS).\n\n"

        "### CRITICAL PENALTIES (Apply these ruthlessly):\n"
        "A. Entropic Penalty: MBH catalysts must be highly rigid and compact molecular bullets to organize the transition state. Heavily penalize floppy molecules or long alkyl chains.\n"
        "B. Chemical Realism: Penalize structures with bizarre, unstable, or competing functional groups that look like AI hallucinations rather than synthesizable, stable lab chemicals.\n\n"

        "### Calibration & Benchmarking:\n"
        "To ensure consistency, use these benchmarks to set your scoring scale:\n"
        "- DABCO (1,4-diazabicyclo[2.2.2]octane): 50.0 (The reference catalyst).\n\n"

        "### Task\n"
        "Analyze each given SMILES against the strict requirements for nucleophilicity, sterics, rigidity, and synthetic realism.\n"
        "Predict its expected catalytic performance (factoring in reaction rate and yield) "
        "on a continuous scale from 0.0 (completely inactive/poor) to 100.0 (highly active/excellent).\n\n"

        "### Output Instructions\n"
        "Return ONLY a valid JSON object where keys are the indices provided and values are the floating-point scores. "
        "Do not include markdown formatting (like ```json), reasoning, commentary, or any text outside the JSON object.\n"
        "Example Schema:\n"
        "{\n"
        '  "0": 75.5,\n'
        '  "1": 15.0\n'
        "}\n\n"

        "### Molecules to Evaluate\n"
        "The molecules are given below as a JSON map of index to SMILES:"
    )

    def _build_variable_suffix(self, smiles_map: dict) -> str:
        """The per-batch part of the prompt: the {index: SMILES} JSON map."""
        return "\n" + json.dumps(smiles_map, indent=2)

    def _build_messages(self, smiles_map: dict) -> list:
        """Build the chat messages, splitting the prompt into a cacheable static
        prefix and a variable SMILES suffix. When caching is on for an Anthropic
        model, an explicit `cache_control` breakpoint is placed on the static
        prefix so the follow-up calls read it from cache instead of re-billing it."""
        variable_suffix = self._build_variable_suffix(smiles_map)

        cache_eligible = self.use_prompt_caching and (
            self.is_anthropic or "anthropic" in self.model_name.lower() or "claude" in self.model_name.lower()
        )
        if cache_eligible:
            return [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._STATIC_PROMPT_PREFIX,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": variable_suffix},
                ],
            }]

        # Non-Anthropic providers cache the identical leading prefix automatically;
        # a single concatenated string keeps that prefix byte-stable across batches.
        return [{"role": "user", "content": self._STATIC_PROMPT_PREFIX + variable_suffix}]

    def _attempt_call(self, messages: list, expected_indices: list) -> dict | None:
        """One LLM attempt. Returns {index: clamped_score} on success, or None if
        the call timed out / errored / returned unparseable content (so the caller
        can decide whether to retry, split, or fall back)."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            # Hard per-call timeout so a stuck request is aborted (then handled).
            "timeout": self.request_timeout,
        }
        if self.is_anthropic:
            kwargs["max_tokens"] = self.max_tokens
        else:
            kwargs["response_format"] = {"type": "json_object"}

        # Bound the thinking budget for heavy reasoning models so they can't
        # over-think forever. OpenRouter accepts a `reasoning` block via
        # extra_body; we only send it when explicitly configured.
        reasoning = {}
        if self.reasoning_effort:
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_max_tokens:
            reasoning["max_tokens"] = int(self.reasoning_max_tokens)
        if reasoning and not self.is_anthropic:
            kwargs["extra_body"] = {"reasoning": reasoning}

        try:
            response = self.client.chat.completions.create(**kwargs)
            raw_scores = json.loads(_extract_json(response.choices[0].message.content))
        except Exception as e:  # noqa: BLE001 - timeout/network/parse: let caller handle it
            print(f"[MBH oracle] Call failed/timed out: {e}")
            return None

        return {int(idx): self._clamp_score(raw_scores.get(str(idx))) for idx in expected_indices}

    @staticmethod
    def _clamp_score(value) -> float:
        """Coerce a raw model score to a float in [0, 100]; unparseable -> 0.0."""
        try:
            return max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _make_batch_api_call(self, smiles_map: dict) -> dict:
        """Score one batch via a fallback ladder, returning {index: score_0_100}.

        Ladder (a slow reasoning answer must never be mistaken for a 0):
          1. Ask the FULL batch up to ``full_batch_attempts`` times.
          2. If all full-batch attempts fail (e.g. repeated timeouts), score each
             molecule INDIVIDUALLY -- single molecules are fast and reasoning-light.
          3. Only a molecule whose individual call also fails is scored 0.0.
        """
        expected_indices = list(smiles_map.keys())
        messages = self._build_messages(smiles_map)

        # Step 1: full batch, up to N attempts.
        for attempt in range(self.full_batch_attempts):
            result = self._attempt_call(messages, expected_indices)
            if result is not None:
                return result
            print(
                f"[MBH oracle] Full-batch attempt {attempt + 1}/{self.full_batch_attempts} "
                f"failed for {len(expected_indices)} molecules."
            )

        # Step 2: fall back to scoring each molecule on its own.
        print(
            f"[MBH oracle] Falling back to per-molecule scoring for "
            f"{len(expected_indices)} molecules."
        )
        scores = {}
        for idx in expected_indices:
            single_msgs = self._build_messages({idx: smiles_map[idx]})
            single = self._attempt_call(single_msgs, [idx])
            if single is not None:
                scores[idx] = single[idx]
            else:
                # Step 3: even the single-molecule call failed -> 0.0.
                print(f"[MBH oracle] Per-molecule call failed for index {idx}; scoring 0.0.")
                scores[idx] = 0.0
        return scores

    def __call__(self, mols: list) -> np.ndarray:
        """Score a list of RDKit Mols. Returns raw scores in [0, 100].

        Saturn applies the reward-shaping sigmoid (low=40, high=90 in the config)
        to these raw values, so we must return the 0-100 scale, not a 0-1 reward.
        """
        results = np.zeros(len(mols), dtype=np.float32)

        all_smiles = []
        valid_indices = []
        for i, mol in enumerate(mols):
            if is_valid_mbh_catalyst(mol, self.max_mol_wt, self.max_rotatable_bonds, self.require_neutral):
                all_smiles.append(Chem.MolToSmiles(mol))
                valid_indices.append(i)
            else:
                all_smiles.append(None)  # invalid -> stays 0.0, not sent to the LLM

        for start in range(0, len(valid_indices), self.batch_size):
            batch_indices = valid_indices[start:start + self.batch_size]
            smiles_map = {idx: all_smiles[idx] for idx in batch_indices}

            if self.use_prompt_caching and self.num_calls > 1:
                # Prime-then-fan-out: run one replicate first so the provider-side
                # cache is warm, then fan out the remaining identical calls as
                # cache hits. (Concurrent calls would all miss the cold cache.)
                batch_runs = [self._make_batch_api_call(smiles_map)]
                if self.num_calls > 2:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_calls - 1) as executor:
                        futures = [
                            executor.submit(self._make_batch_api_call, smiles_map)
                            for _ in range(self.num_calls - 1)
                        ]
                        batch_runs.extend(f.result() for f in futures)
                else:
                    batch_runs.append(self._make_batch_api_call(smiles_map))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_calls) as executor:
                    futures = [
                        executor.submit(self._make_batch_api_call, smiles_map)
                        for _ in range(self.num_calls)
                    ]
                    batch_runs = [f.result() for f in futures]

            for idx in batch_indices:
                per_call = [run[idx] for run in batch_runs]
                results[idx] = float(sum(per_call) / len(per_call))

            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return results
