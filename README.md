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

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MiquelAngelPerezPuigdo/Agentic-AI-for-organocatalyst-design.git](https://github.com/MiquelAngelPerezPuigdo/Agentic-AI-for-organocatalyst-design.git)
   cd Agentic-AI-for-organocatalyst-design
   ```

2. **Create the environment:**
   Use the provided YAML file to build the specific dependencies.
   ```bash
   conda env create -f env.yml
   conda activate ppchem
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API key:
   ```env
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

4. **Prepare Data:**
   Ensure your reference PDF papers are located in a `publications/` directory in the root folder. The benchmark uses PyMuPDF to extract context directly from these files.

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

### 3. Run Prospective Case (Active Learning)
```bash
python main.py --mode prospective
```

---

## 📂 Project Structure

```text
Agentic-AI-for-organocatalyst-design/
├── main.py              # Master switchboard / Entry point
├── config.py            # API keys and model configurations
├── env.yml              # Conda environment dependencies
├── publications/        # Folder containing source PDFs
├── output/              # Generated results, plots, and CSV logs
└── src/                 
    ├── api.py           # Robust OpenRouter API handling logic
    ├── document.py      # PDF text extraction and cleaning
    ├── evaluation.py    # Metric calculation (Judge LLM, JSON parsers)
    ├── ligand_data.py   # Ground truth SMILES, metadata, and seed data
    ├── prompts.py       # Prompt templates for all 3 benchmarks
    ├── report.py        # Seaborn/Matplotlib visualization generation
    ├── runner.py        # Multithreaded parallel execution engine
    └── surrogate.py     # ML Hybrid Surrogate Model (RDKit / Scikit-Learn)
```

## 📊 Outputs

All benchmarks automatically output data into the `output/` directory:
* **CSV Files:** Raw performance metrics and LLM responses for downstream analysis.
* **Plots:** High-quality `.png` visualizations (e.g., `benchmark2_trend.png`, `generative_active_learning.png`).
* **Markdown Reports:** Formatted summaries of the LLM Judge assessments.

---

## ⚖️ License

MIT License

Copyright (c) 2026 Miquel Angel Perez Puigdomenech

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
