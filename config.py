import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Models
FRONTIER_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-5.4-pro",
    "google/gemini-3.1-pro-preview",
    "xiaomi/mimo-v2-pro"
]

OPEN_MODELS = [
    "meta-llama/llama-3.1-70b-instruct",
    "deepseek/deepseek-r1",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen3.5-122b-a10b"
]

ALL_MODELS = FRONTIER_MODELS + OPEN_MODELS
JUDGE_MODEL = "google/gemini-3.1-pro-preview"
ITERATIONS = 5 

# Reaction Data (Make sure your PDFs are inside the "data/" folder)
REACTIONS = {
    "MBH_Tertiary_Amine": {
        "name": "Morita-Baylis-Hillman (MBH) catalyzed with a tertiary amine",
        "file": "data/recent-advances-in-catalytic-systems-for-the-mechanistically-complex-morita-baylis-hillman-reaction.pdf"
    },
    "Pd_Desulfonylative_Fluorination": {
        "name": "Pd-catalyzed desulfonylative fluorination of electron-deficient (hetero)aryl sulfonyl fluorides",
        "file": "data/Pd-catalyzed desulfonylative fluorination of electron-deficient (hetero)aryl sulfonyl fluorides.pdf"
    },
    "Ni_Photoredox_Cross_Electrophile": {
        "name": "Ni/Photoredox-Catalyzed Enantioselective Cross-Electrophile Coupling of Styrene Oxides with Aryl Iodides",
        "file": "data/ni-photoredox-catalyzed-enantioselective-cross-electrophile-coupling-of-styrene-oxides-with-aryl-iodides.pdf"
    }, 
    "Cu_Radical_N_Alkylation": {
        "name": "Copper-catalyzed C(sp3)-N cross-coupling of alkylboronic pinacol esters (Bpin) with N-nucleophiles via single-electron transfer (SET) oxidation and radical boron group abstraction",
        "file": "data/radical-strategy-to-the-boron-to-copper-transmetalation-problem-n-alkylation-with-alkylboronic-esters.pdf"
}
}