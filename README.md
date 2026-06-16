# Agentic AI for Catalyst Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2025.9.6-green.svg)](https://www.rdkit.org/)

A modular, automated evaluation framework for benchmarking Large Language Models (LLMs) on complex chemical tasks. This repository investigates the capacity of frontier AI models to reason through reaction mechanisms, rank catalytic ligands, and autonomously design novel organocatalysts through active learning.

## 🚀 Capabilities

This suite is divided into three distinct benchmarking modules:

1. **Mechanistic Understanding (Benchmark 1):** Evaluates an LLM's ability to reason through reaction mechanisms and deduce favorable catalyst properties. It contrasts zero-shot chemical knowledge against context-augmented reasoning using scientific literature (PDF parsing). Scored via an LLM-as-a-Judge approach.

2. **Ligand Ranking (Benchmark 2):** Tests the model's ability to rank catalytic ligands by performance across various reaction classes (e.g., Pd-Fluorination, Ni-Epoxide Coupling). It maps performance across 6 levels of prompting complexity, ranging from basic zero-shot to Chain-of-Thought (CoT) and JSON-enforced instruction following. Evaluated using the Top-5 Overlap metric.

3. **Active Learning (Prospective Case):** Implements an iterative, agentic AI discovery campaign. The LLM acts as a generative agent proposing novel ligands to maximize reaction yield. Experimental feedback is simulated using a Hybrid ML Surrogate Model (Random Forest + Ridge Regression mapping RDKit Morgan Fingerprints, LogP, and TPSA). 

---

## 🛠 Prerequisites

* **Conda/Miniconda:** Required to manage the chemistry-specific dependencies (`rdkit`, `scikit-learn`).
* **OpenRouter API Key:** The suite interfaces with frontier models (e.g., Gemini, Claude, GPT) dynamically via OpenRouter.
* **Anthropic API Key (optional, for `saturn-mbh`):** The MBH oracle can score directly against Anthropic's API (e.g. Claude Opus) by setting `ANTHROPIC_API_KEY`; see [Saturn-MBH usage](#4-saturn-mbh-de-novo-mbh-catalyst-design-via-generative-model).
* **Saturn (only for `--mode saturn-mbh`):** The *de novo* MBH catalyst-design campaign drives an external [Saturn](https://github.com/MiquelAngelPerezPuigdo/saturn) generative model. All other modules run without it. See [Saturn setup](#-saturn-setup-only-for---mode-saturn-mbh) below.

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MiquelAngelPerezPuigdo/Agentic-AI-for-Catalyst-Design.git
   cd Agentic-AI-for-Catalyst-Design
   ```

2. **Create the environment:**
   Use the provided YAML file to build the specific dependencies.
   ```bash
   conda env create -f env.yml
   conda activate ppchem
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API key(s):
   ```env
   OPENROUTER_API_KEY=your_actual_api_key_here
   # Optional: for scoring saturn-mbh candidates directly with Claude
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

4. **Prepare Data:**
   Ensure your reference PDF papers are located in a `publications/` directory in the root folder. The benchmark uses PyMuPDF to extract context directly from these files.

---

## 🪐 Saturn Setup (only for `--mode saturn-mbh`)

The `saturn-mbh` mode is the only part of this suite that needs an external dependency: the [Saturn](https://github.com/MiquelAngelPerezPuigdo/saturn) sample-efficient generative model (this is a fork of [schwallergroup/saturn](https://github.com/schwallergroup/saturn) carrying the MBH oracle registration). Skip this section entirely if you are not running `saturn-mbh`.

1. **Clone Saturn** somewhere outside this repository:
   ```bash
   git clone https://github.com/MiquelAngelPerezPuigdo/saturn.git ~/saturn
   ```

2. **Create Saturn's environment** (separate from `ppchem`; see Saturn's own README for troubleshooting):
   ```bash
   cd ~/saturn
   source setup.sh   # creates and configures the 'saturn' conda env
   ```

3. **Tell this repo where Saturn lives.** The launcher looks for it at, in order: the `--saturn-home` flag, the `SATURN_HOME` environment variable, then `~/saturn` (the default). To use a custom location:
   ```bash
   export SATURN_HOME=/path/to/saturn
   ```

At launch, the bridge copies this project's MBH oracle into the Saturn checkout (as `agentic_mbh_oracle.py`, registered as the component `agentic_mbh_catalyst_score`) and runs Saturn in its own conda env — it never modifies this repository's environment. See the [Saturn-MBH usage section](#4-saturn-mbh-de-novo-mbh-catalyst-design-via-generative-model) for run commands.

---

## 🖥 Usage

The suite utilizes `main.py` as a master switchboard to execute different benchmarking modules. Use the `--mode` argument to select your target task.

### 1. Run Benchmark 1 (Mechanistic Understanding)
```bash
python main.py --mode benchmark1
```

### 2. Run Benchmark 2 (Ligand Ranking)
```bash
python main.py --mode benchmark2
```

### 3. Run Prospective Case (Active Learning & Ablation Study)
```bash
python main.py --mode prospective --ablation A --campaigns 5 --steps 14
```

#### 🔬 Ablation Analysis Options
The prospective active learning module includes a rigorous ablation suite to isolate which structural, policy, or domain knowledge features contribute to the LLM's design capabilities.

You can select an ablation mode using the `--ablation` argument:

All ablations share the same robust baseline configuration so that each removed feature is isolated fairly. Every mode uses the same standard, unbiased optimization prompt (no exploration/exploitation portfolio directives).

| Mode | Name | SMILES Representation | History Order | Chemical Domain Info | Scientific Purpose |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **`A`** | **Baseline** | Shuffled (`doRandom=True`) | Sorted (SAR Ladder) | Full (Context + Mechanism) | Unbiased reference configuration to measure chemical optimization power. |
| **`B`** | **`-SAR Ladder`** | Shuffled (`doRandom=True`) | Chronological (Insertion) | Full (Context + Mechanism) | Evaluates if sorting the history by yield is critical for optimization. |
| **`C`** | **`-SMILES Shuffle`** | Canonical (Deterministic) | Sorted (SAR Ladder) | Full (Context + Mechanism) | Evaluates if deterministic string/prefix matching introduces local minima bias. |
| **`D`** | **`-Mechanism`** | Shuffled (`doRandom=True`) | Sorted (SAR Ladder) | Partial (Context only) | Evaluates if specific reaction mechanism guidelines are useful. |
| **`E`** | **`Chemistry Agnostic`** | Shuffled (`doRandom=True`) | Sorted (SAR Ladder) | None (Blind/Blackout) | Evaluates if the LLM can optimize purely on molecular structure representations. |
| **`F`** | **`Full Ablation`** | Canonical (Deterministic) | Chronological (Insertion) | None (Blind/Blackout) | Purely blind tabular optimization over deterministic text string records. |
| **`all`** | **`All Modes`** | Runs all modes (A through F) sequentially and generates high-quality comparison plots. |

### 🧪 Real-World Constrained Campaigns (Synthetic & Operational Feasibility)

To showcase how the pipeline can adapt "by a stroke of prompt" without retraining models or altering the underlying machine learning surrogates, we introduce three realistic laboratory constraints for prospective campaigns. These reflect actual operational limitations chemists face daily:

#### 1. High-Throughput Batching ($6 \times 7$)
* **Scientific Purpose:** Explores the efficiency of *experimental throughput* (broad batch optimization) under a fixed laboratory assay budget of 42 reactions. This run is set up to propose a wide batch of $6$ ligands per step over $7$ steps, allowing direct performance comparison against the deep sequential baseline of Ablation A ($3$ ligands per step over $14$ steps).
* **Operational Setup:** Configures the generator to propose exactly $6$ unique ligands per step over a compressed timeline of $7$ iterations. This evaluates whether the model is more effective at optimizing with broader parallel context per step rather than more frequent feedback cycles.

#### 2. Divergent Synthetic Accessibility (Single Common Intermediate)
* **Scientific Purpose:** Reduces synthetic overhead by forcing the LLM to act like a practical organic chemist.
* **Operational Setup:** Each round, the LLM is prompted **and enforced** (proposals are rejected and re-requested) to ensure all ligands in the batch share a common building block accessible via a single divergent synthetic pathway. To avoid trapping the search short of the global optimum, the constraint is **automatically released once the campaign reaches a strong lead (best yield ≥ 80%)**, freeing the agent to make the final move needed to reach the 89% champion.
* **Enforcement audit:** A deterministic RDKit audit (counted only over constraint-active steps) quantifies how faithfully the constraint was honored using **three complementary metrics** plus a binary PASS verdict: (i) **MCS conservation** — fraction of each molecule's heavy atoms in the batch's Maximum Common Substructure core (≥0.5 to pass); (ii) **Bemis–Murcko scaffold agreement** — fraction of the batch sharing the same generic ring scaffold (=1.0, primary gate); and (iii) **mean pairwise Tanimoto** similarity (Morgan fingerprints, ≥0.4). A campaign suite is reported as *divergence-enforced* when ≥80% of active batches pass.

#### 3. Self-Driving-Lab Click Library (Combinatorial Building-Block Search)
* **Scientific Purpose:** Mirrors a real self-driving lab, where the accessible chemical space is defined by an inventory of reagents rather than free-form molecule design.
* **Operational Setup:** Each 1,4-triazole ligand in the master dataset is retrosynthetically split (CuAAC disconnection) into its **alkyne** and **azide** precursors, yielding 18 alkynes + 18 azides. Their full cross-product defines a **324-member virtual library** (a 9× compression: 36 fragments imply 324 molecules). The LLM is fed only the 36 building blocks and proposes compact `(alkyne, azide)` index pairs; the pipeline recombines and scores each product. The unique 89%-yield champion is reachable as one specific pairing (top 0.3% of the library).

**Running the constrained campaigns:**
```bash
# Constraint 1 — High-throughput batching (auto-configured to 6 ligands x 7 steps)
python main.py --mode prospective --constraint batch --campaigns 5 --save-details

# Constraint 2 — Divergent synthesis (enforced + auto-released at 80%, with RDKit audit)
python main.py --mode prospective --constraint divergent --campaigns 5 --steps 14 --save-details

# Constraint 3 — SDL click library (LLM proposes (alkyne, azide) pairs over 36 building blocks)
python main.py --mode prospective --constraint click --campaigns 5 --steps 14 --save-details

# Constraint 3b — SDL click library, chemistry-agnostic (mechanism/context stripped, analog of Ablation E)
python main.py --mode prospective --constraint click --chem-agnostic --campaigns 5 --steps 14 --save-details
```

> The `--chem-agnostic` flag strips all reaction context/mechanism from the click prompt to test whether chemical knowledge helps library *selection* (it accelerates the early search but both variants reach the 89% champion).

> The click library itself can be (re)generated from the master dataset with `python scripts/build_click_library.py`, which writes the alkyne/azide building blocks and the enumerated 324-molecule library to `output/`.

#### Optional CLI Arguments
* `--campaigns <int>`: Number of parallel active learning campaigns to execute (default: `5`).
* `--steps <int>`: Number of iterative optimization steps per campaign (default: `14`).
* `--save-details`: Saves raw LLM responses, proposed SMILES, canonical conversions, and step-by-step progress metrics as detailed JSON logs.

---

### 4. Saturn-MBH (De Novo MBH Catalyst Design via Generative Model)

This mode runs the *de novo* Morita-Baylis-Hillman (MBH) catalyst-design campaign described in the thesis. It couples the **Saturn** sample-efficient generative model (Guo *et al.*) with this project's **MBH LLM-oracle**: Saturn proposes candidate molecules, and the oracle filters them to valid tertiary-amine catalysts (hard structural guards) before batch-scoring the survivors against the MBH mechanism, with DABCO calibrated to 50/100.

Saturn itself lives **outside** this repository. Only the essential MBH delta is brought under this roof: the oracle ([`src/saturn_mbh/mbh_oracle.py`](src/saturn_mbh/mbh_oracle.py)) and a launcher ([`src/saturn_mbh/launcher.py`](src/saturn_mbh/launcher.py)) that injects the oracle into an existing Saturn checkout and calls it.

**Prerequisites:**
* A Saturn checkout (defaults to `~/saturn`; override with `--saturn-home` or `export SATURN_HOME=...`).
* A conda env where Saturn runs (defaults to `saturn`; override with `--saturn-env`).
* An API key for the oracle. The backend is auto-detected from the key:
  * `ANTHROPIC_API_KEY` (`sk-ant-...`) → scores directly via Anthropic's API (e.g. Claude Opus).
  * `OPENROUTER_API_KEY` → scores via OpenRouter (Gemini, GPT, etc.), like the other modules.
  The key is read from the environment and forwarded to the Saturn subprocess; it is **never** written into the config file on disk.

```bash
# Dry run: inject the oracle + write the Saturn config, but DO NOT launch (no API/GPU cost)
python main.py --mode saturn-mbh --dry-run

# Full campaign with the default OpenRouter model (500 oracle-call budget by default)
python main.py --mode saturn-mbh --budget 500

# Run with Claude Opus via Anthropic (set ANTHROPIC_API_KEY first)
python main.py --mode saturn-mbh --model claude-opus-4-8 --budget 500

# Point at a specific Saturn install / env / device
python main.py --mode saturn-mbh --saturn-home ~/saturn --saturn-env saturn --device cuda
```

On launch the bridge: (1) copies the oracle into Saturn as `agentic_mbh_oracle.py` and registers it as the component `agentic_mbh_catalyst_score` (idempotent — never clobbers the fork's own files, and defensively disables any broken sibling oracle import); (2) writes a Saturn config into a **per-model** folder `output/saturn_mbh/run_<model>/` (so different models' artifacts never get mixed up); (3) runs `saturn.py` in the Saturn conda env; (4) renders the score-vs-calls figure from the resulting `oracle_history.csv`.

> Example campaign results are included under [`output/saturn_mbh/`](output/saturn_mbh/): `run_claude-opus-4-8/` (Claude Opus) and `run_gemini_3-5_Flash/` (Gemini), each with its `oracle_history.csv`, `score_vs_calls` figure, and the plot script. The figure can be regenerated at any time with `python output/saturn_mbh/run_<model>/plot_score_vs_calls.py`. Saturn agent checkpoints (`.ckpt`, multi-GB) are intentionally **not** vendored (see `.gitignore`).

#### Optional CLI Arguments (saturn-mbh)
* `--budget <int>`: Oracle-call budget for the campaign (default: `500`).
* `--saturn-home <path>`: Path to the external Saturn checkout (default: `$SATURN_HOME` or `~/saturn`).
* `--saturn-env <name>`: Conda env in which to run Saturn (default: `saturn`).
* `--device <cuda|cpu>`: Device passed to Saturn (default: `cuda`).
* `--model <id>`: Model for the oracle. An OpenRouter id (default: `google/gemini-3.5-flash`) or an Anthropic id (e.g. `claude-opus-4-8`) when `ANTHROPIC_API_KEY` is set.
* `--dry-run`: Inject the oracle and write the config, but do not launch Saturn.

---

## 📂 Project Structure

```text
Agentic-AI-for-Catalyst-Design/
├── main.py                 # Master switchboard / Entry point
├── config.py               # API keys and model configurations
├── env.yml                 # Conda environment dependencies
├── scripts/                # Standalone utility scripts
│   └── build_click_library.py  # Retrosynthetic CuAAC split -> 324-member click library
├── publications/           # Folder containing source PDFs
├── output/                 # Generated results, plots, and CSV logs (see subfolders below)
└── src/                 
    ├── api.py              # Robust OpenRouter API handling logic
    ├── ablation.py         # Ablation + constrained-campaign execution and plotting
    ├── click_library.py    # Click building blocks & (alkyne, azide) -> product recombination
    ├── document.py         # PDF text extraction and cleaning
    ├── evaluation.py       # Metric calculation (Judge LLM, JSON parsers)
    ├── ligand_data.py      # Ground truth SMILES, metadata, and seed data
    ├── prompts.py          # Prompt templates for all benchmarks & campaigns
    ├── report.py           # Seaborn/Matplotlib visualization generation
    ├── runner.py           # Multithreaded parallel execution engine
    ├── surrogate.py        # ML Hybrid Surrogate Model (RDKit / Scikit-Learn)
    └── saturn_mbh/         # MBH de novo design: MBH LLM-oracle + launcher that calls external Saturn
        ├── mbh_oracle.py           # Tertiary-amine filters + MBH-mechanism LLM scoring (DABCO=50)
        ├── launcher.py             # Injects the oracle into a Saturn checkout and runs it
        └── plot_score_vs_calls.py  # Score-vs-oracle-calls figure (auto-detects CSV/columns)
```

## 📊 Outputs

All benchmarks automatically output data into the `output/` directory:
* **CSV Files:** Raw performance metrics and LLM responses for downstream analysis.
* **Plots:** High-quality `.png` visualizations (e.g., `benchmark2_trend.png`, `generative_active_learning.png`).
* **Markdown Reports:** Formatted summaries of the LLM Judge assessments.

---

## 🙏 Acknowledgments & Citation

I am grateful to **Sarina Kopf** and **Prof. Philippe Schwaller** for their supervision, guidance, and for the opportunity to carry out this work.

The `saturn-mbh` *de novo* design loop is powered by **Saturn**, the sample-efficient generative molecular design model by **Jeff Guo, Junwu Chen, Anthony GX-Chen, and Philippe Schwaller**. The generator itself is entirely their work — this repository only contributes the MBH-specific LLM oracle and the bridge that drives Saturn. Huge thanks to Jeff for building and open-sourcing such a capable molecule generator.

* **Saturn:** <https://github.com/schwallergroup/saturn>
* **Publication:** Guo, J.; Chen, J.; GX-Chen, A.; Schwaller, P. *Sample-efficient generative molecular design using memory manipulation.* **Nature Machine Intelligence** **2026**, *8* (3), 449–460.

If you use the `saturn-mbh` mode, please cite Saturn:

```bibtex
@article{Guo2026,
  author  = {Guo, Jeff and Chen, Junwu and GX-Chen, Anthony and Schwaller, Philippe},
  title   = {Sample-efficient generative molecular design using memory manipulation},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  volume  = {8},
  number  = {3},
  pages   = {449--460}
}
```

---

## ⚖️ License

MIT License

Copyright (c) 2026 Miquel Angel Perez Puigdomenech

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
