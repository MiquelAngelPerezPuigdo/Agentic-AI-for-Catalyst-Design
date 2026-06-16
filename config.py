import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Models
FRONTIER_MODELS = [
    "anthropic/claude-opus-4.7",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.3"
]

OPEN_MODELS = [
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.6-plus"
]

ALL_MODELS = FRONTIER_MODELS + OPEN_MODELS
JUDGE_MODEL = "google/gemini-3.1-pro-preview"
ITERATIONS = 5 

# Prompt caching: when True, the benchmark runners "prime" the shared prompt prefix
# with a single serial call per (combination) before fanning out the remaining
# identical iterations in parallel, so they hit a warm provider-side cache.
# Also enables explicit Anthropic `cache_control` breakpoints on the large static block.
USE_PROMPT_CACHING = True

# Publication Data ("publications/" folder)
REACTIONS = {
    "MBH_Tertiary_Amine": {
        "name": "Morita-Baylis-Hillman (MBH) catalyzed with a tertiary amine",
        "file": "publications/recent-advances-in-catalytic-systems-for-the-mechanistically-complex-morita-baylis-hillman-reaction.pdf"
    },
    "Pd_Desulfonylative_Fluorination": {
        "name": "Pd-catalyzed desulfonylative fluorination of electron-deficient (hetero)aryl sulfonyl fluorides",
        "file": "publications/Pd-catalyzed desulfonylative fluorination of electron-deficient (hetero)aryl sulfonyl fluorides.pdf"
   },
    "Ni_Photoredox_Cross_Electrophile": {
        "name": "Ni/Photoredox-Catalyzed Enantioselective Cross-Electrophile Coupling of Styrene Oxides with Aryl Iodides",
        "file": "publications/ni-photoredox-catalyzed-enantioselective-cross-electrophile-coupling-of-styrene-oxides-with-aryl-iodides.pdf"
    },
    "Cu_Radical_N_Alkylation": {
        "name": "Copper-catalyzed C(sp3)-N cross-coupling of alkylboronic pinacol esters (Bpin) with N-nucleophiles via single-electron transfer (SET) oxidation and radical boron group abstraction",
        "file": "publications/radical-strategy-to-the-boron-to-copper-transmetalation-problem-n-alkylation-with-alkylboronic-esters.pdf"
    },
    "Pd_Catalyzed_Methylene_beta_CH_Fluorination_Native_Amides": {
        "name": "Palladium-Catalyzed Methylene betaCH Fluorination of Native Amides",
        "file": "publications/palladium-catalyzed-methylene-beta-c-h-fluorination-of-native-amides.pdf"
    },
    "Dual-Ligand System for Mild Decarbonylative Suzuki–Miyaura Cross-Coupling of Aroyl Chlorides": {
        "name": "Dual-Ligand System for Mild Decarbonylative Suzuki–Miyaura Cross-Coupling of Aroyl Chlorides",
        "file": "publications/dual-ligand-system-for-mild-decarbonylative-suzuki-miyaura-cross-coupling-of-aroyl-chlorides.pdf"
    }
}