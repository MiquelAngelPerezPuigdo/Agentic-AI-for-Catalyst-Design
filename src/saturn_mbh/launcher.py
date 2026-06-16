"""
Bridge that runs an MBH *de novo* catalyst-design campaign by calling an external
Saturn install (Jeff Guo's sample-efficient generative model) with this project's
MBH LLM-oracle.

Design intent: keep Saturn itself *outside* this repository. We only bring the
essential MBH delta under this roof (``mbh_oracle.py``) and "call Saturn to do the
deed". At launch we:

  1. Locate the Saturn checkout (``SATURN_HOME`` env var, else ~/saturn).
  2. Inject this repo's MBH oracle into that checkout under a distinct name
     (``agentic_mbh_catalyst_score``) so we never clobber the fork's own files,
     and register it in Saturn's ``oracles/utils.py`` (idempotently).
  3. Render a Saturn JSON config into ``output/saturn_mbh/`` from a template,
     filling in the prior, budget, batch size, model and API key.
  4. Run ``saturn.py <config>`` in the Saturn conda env as a subprocess.

Nothing here imports Saturn into this process; Saturn runs in its own env.
"""

import os
import json
import shutil
import subprocess
from pathlib import Path

# Repo root = three levels up from this file (src/saturn_mbh/launcher.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_SRC = Path(__file__).resolve().parent / "mbh_oracle.py"

# Name under which we register/inject our oracle inside Saturn (avoids clobbering
# the fork's existing MBH_catalyst_score.py).
INJECTED_MODULE = "agentic_mbh_oracle"
INJECTED_COMPONENT = "agentic_mbh_catalyst_score"
INJECTED_CLASS = "MBHcatalystscore"

DEFAULT_PRIOR = "experimental_reproduction/checkpoint_models/zinc-250k-aizynth-purged-epoch-100.prior"


def resolve_saturn_home(saturn_home: str | None = None) -> Path:
    """Find the Saturn checkout. Priority: explicit arg > $SATURN_HOME > ~/saturn."""
    candidate = saturn_home or os.getenv("SATURN_HOME") or str(Path.home() / "saturn")
    path = Path(candidate).expanduser().resolve()
    if not (path / "saturn.py").exists():
        raise FileNotFoundError(
            f"Could not find a Saturn install at '{path}' (no saturn.py). "
            f"Set SATURN_HOME to your Saturn checkout, e.g. export SATURN_HOME=~/saturn"
        )
    return path


def _comment_out_missing_similarity_imports(saturn_home: Path, text: str) -> str:
    """Defensively comment out 'from oracles.similarity.X import ...' lines whose
    module file X.py is absent. A partially-staged fork (e.g. a Butenolide import
    with no Butenolide file) would otherwise crash oracles/utils.py at import time
    and block the MBH campaign. We never delete or touch existing files."""
    sim_dir = saturn_home / "oracles" / "similarity"
    out_lines = []
    for ln in text.splitlines(keepends=True):
        stripped = ln.lstrip()
        if stripped.startswith("from oracles.similarity.") and not stripped.startswith("#"):
            module = stripped.split("from oracles.similarity.", 1)[1].split(" ", 1)[0].strip()
            if module != INJECTED_MODULE and not (sim_dir / f"{module}.py").exists():
                out_lines.append(f"# [agentic-mbh disabled: missing module] {ln}")
                continue
        out_lines.append(ln)
    return "".join(out_lines)


def inject_oracle(saturn_home: Path) -> None:
    """Copy our oracle into Saturn and register it in oracles/utils.py (idempotent)."""
    # 1. Copy the oracle module into Saturn's similarity oracles folder.
    dest = saturn_home / "oracles" / "similarity" / f"{INJECTED_MODULE}.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ORACLE_SRC, dest)

    # 2. Register it in oracles/utils.py if not already present.
    utils_path = saturn_home / "oracles" / "utils.py"
    text = utils_path.read_text()

    # 2a. Neutralize any broken sibling imports (e.g. a missing Butenolide module).
    text = _comment_out_missing_similarity_imports(saturn_home, text)

    import_line = (
        f"from oracles.similarity.{INJECTED_MODULE} import {INJECTED_CLASS} "
        f"as _AgenticMBH\n"
    )
    if INJECTED_MODULE not in text:
        # Insert the import after the first 'from oracles.similarity' import.
        lines = text.splitlines(keepends=True)
        anchor = next(
            (i for i, ln in enumerate(lines) if ln.startswith("from oracles.similarity")),
            0,
        )
        lines.insert(anchor + 1, import_line)
        text = "".join(lines)

    if f'name == "{INJECTED_COMPONENT}"' not in text:
        # Add a dispatch branch right after the function's opening so it wins early.
        marker = "    name = oracle_component_parameters.name\n"
        branch = (
            f'    if name == "{INJECTED_COMPONENT}":\n'
            f"        return _AgenticMBH(oracle_component_parameters)\n"
        )
        if marker in text:
            text = text.replace(marker, marker + branch, 1)
        else:
            raise RuntimeError(
                "Could not find the dispatch anchor in Saturn's oracles/utils.py; "
                "Saturn's structure may have changed."
            )

    utils_path.write_text(text)


def render_config(
    saturn_home: Path,
    run_dir: Path,
    api_key: str,
    model: str,
    budget: int,
    batch_size: int,
    num_calls: int,
    prior: str,
    seed: int,
    device: str,
) -> Path:
    """Write a Saturn goal-directed-generation config JSON for the MBH campaign."""
    prior_path = str((saturn_home / prior).resolve())
    config = {
        "logging": {
            "logging_frequency": 100,
            "logging_path": str(run_dir / "log.log"),
            "model_checkpoints_dir": str(run_dir),
        },
        "oracle": {
            "budget": budget,
            "allow_oracle_repeats": False,
            "aggregator": "product",
            "components": [
                {
                    "name": INJECTED_COMPONENT,
                    "weight": 1,
                    "preliminary_check": False,
                    "specific_parameters": {
                        "api_key": api_key,
                        "model_name": model,
                        "num_calls": num_calls,
                        "batch_size": batch_size,
                        "rate_limit_delay": 1.0,
                    },
                    "reward_shaping_function_parameters": {
                        "transformation_function": "sigmoid",
                        "parameters": {"low": 40, "high": 90, "k": 0.15},
                    },
                },
                {
                    "name": "mw",
                    "weight": 1,
                    "preliminary_check": False,
                    "specific_parameters": {},
                    "reward_shaping_function_parameters": {
                        "transformation_function": "double_sigmoid",
                        "parameters": {
                            "low": 50,
                            "high": 400,
                            "coef_div": 500,
                            "coef_si": 250,
                            "coef_se": 250,
                        },
                    },
                },
            ],
        },
        "goal_directed_generation": {
            "reinforcement_learning": {
                "prior": prior_path,
                "agent": prior_path,
                "batch_size": batch_size,
                "learning_rate": 0.0001,
                "sigma": 128.0,
                "augmented_memory": True,
                "augmentation_rounds": 10,
                "selective_memory_purge": True,
            },
            "experience_replay": {"memory_size": 100, "sample_size": 10, "smiles": []},
            "diversity_filter": {"name": "IdenticalMurckoScaffold", "bucket_size": 10},
            "hallucinated_memory": {
                "execute_hallucinated_memory": False,
                "hallucination_method": "ga",
                "num_hallucinations": 100,
                "num_selected": 5,
                "selection_criterion": "random",
            },
            "beam_enumeration": {
                "execute_beam_enumeration": False,
                "beam_k": 2,
                "beam_steps": 18,
                "substructure_type": "structure",
                "structure_min_size": 15,
                "pool_size": 4,
                "pool_saving_frequency": 1000,
                "patience": 5,
                "token_sampling_method": "topk",
                "filter_patience_limit": 100000,
            },
        },
        "running_mode": "goal_directed_generation",
        "model_architecture": "mamba",
        "device": device,
        "seed": seed,
    }
    config_path = run_dir / "mbh_saturn_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def run_mbh_campaign(
    *,
    model: str = "google/gemini-3.5-flash",
    budget: int = 500,
    batch_size: int = 16,
    num_calls: int = 3,
    saturn_home: str | None = None,
    saturn_env: str = "saturn",
    prior: str = DEFAULT_PRIOR,
    seed: int = 0,
    device: str = "cuda",
    api_key: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Inject the MBH oracle into Saturn and launch a de novo MBH design campaign.

    Returns the run directory under output/saturn_mbh/ holding the config, logs and
    Saturn checkpoints. With dry_run=True it injects + writes the config but does not
    launch Saturn (useful for testing without spending API budget or GPU time).
    """
    from src.paths import out_dir  # local import to avoid hard dep at module load

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key and not dry_run:
        raise SystemExit(
            "[!] OPENROUTER_API_KEY is not set. Add it to your .env or export it before launching."
        )

    saturn_path = resolve_saturn_home(saturn_home)
    inject_oracle(saturn_path)

    # Canonical run dir, matching the results already brought under this repo's roof.
    run_dir = Path(out_dir("saturn_mbh")) / "mbh_catalyst_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = render_config(
        saturn_home=saturn_path,
        run_dir=run_dir,
        api_key=api_key or "DRY_RUN_NO_KEY",
        model=model,
        budget=budget,
        batch_size=batch_size,
        num_calls=num_calls,
        prior=prior,
        seed=seed,
        device=device,
    )

    print(f"[+] Saturn home:   {saturn_path}")
    print(f"[+] MBH oracle injected as component '{INJECTED_COMPONENT}'.")
    print(f"[+] Config written: {config_path}")

    if dry_run:
        print("[i] Dry run: skipping Saturn launch.")
        return run_dir

    cmd = [
        "conda", "run", "--no-capture-output", "-n", saturn_env,
        "python", "saturn.py", str(config_path),
    ]
    print(f"[+] Launching Saturn (env '{saturn_env}'):\n    {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(saturn_path), check=True)

    # Render the score-vs-calls figure from the campaign's oracle_history.
    _generate_plot(run_dir)

    print(f"[+] Campaign finished. Outputs in {run_dir}")
    return run_dir


def _generate_plot(run_dir: Path) -> None:
    """Run the bundled score-vs-calls plot script against this run's oracle_history."""
    plot_script = run_dir / "plot_score_vs_calls.py"
    if not plot_script.exists():
        bundled = Path(__file__).resolve().parent / "plot_score_vs_calls.py"
        if bundled.exists():
            shutil.copyfile(bundled, plot_script)
        else:
            print("[i] No plot script found; skipping figure generation.")
            return
    try:
        subprocess.run(["python", str(plot_script)], cwd=str(run_dir), check=True)
        print(f"[+] Score-vs-calls figure written in {run_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[!] Plot generation failed ({e}); the raw oracle_history CSV is still available.")
