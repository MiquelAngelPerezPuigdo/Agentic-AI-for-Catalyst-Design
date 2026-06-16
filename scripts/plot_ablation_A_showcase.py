"""Thesis showcase figure: Ablation A trajectory with the real database catalysts
the LLM re-discovered drawn next to the step at which it proposed them.

Narrative: the agent is run with NO access to the ground-truth dataset, yet several of
its de-novo proposals canonicalize EXACTLY onto known high-performing catalysts in the
master database. We highlight the record-setting hits (each one a new global best) and
render their 2D structures, connected with a leader line to the point on the optimization
curve where the agent first proposed them. This visualises the chemical knowledge the LLM
brought to bear: it walked the SAR ladder straight onto the literature's best ligands.

Run:
    conda run -n ppchem python scripts/plot_ablation_A_showcase.py [campaign_id]
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from src.surrogate import init_surrogate
from src.paths import out_path

START_YIELD = 37.0
ACCENT = "#4285F4"   # Ablation A blue (matches STYLE_CONFIGS)
HIT_COLOR = "#16a085"


def mol_to_image(smiles, size=(330, 230)):
    """Render a molecule to an RGBA numpy array with a transparent background."""
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    png = d.GetDrawingText()
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    return np.asarray(img)


def build_figure(campaign_id=5):
    details_path = out_path("campaign_details", f"campaign_details_A_campaign_{campaign_id}.json")
    with open(details_path) as f:
        logs = json.load(f)

    _, _, dataset_dict = init_surrogate()

    # Per-step max yield + identify record-setting DB-matched proposals.
    steps, max_curve = [], []
    gmax = 0.0
    record_hits = []   # (step, yield, canonical_smiles, db_yield_str)
    for s in logs:
        step = s["step"]
        step_yields = [p["yield"] for p in s["proposals"]]
        step_max = max(step_yields)
        # best DB-matched proposal this step
        db_best = None
        for p in s["proposals"]:
            if p["canonical_smiles"] in dataset_dict:
                if db_best is None or p["yield"] > db_best[1]:
                    db_best = (step, p["yield"], p["canonical_smiles"], dataset_dict[p["canonical_smiles"]])
        if step_max > gmax + 1e-9 and db_best is not None and db_best[1] >= step_max - 1e-6:
            record_hits.append(db_best)
        gmax = max(gmax, step_max)
        steps.append(step)
        max_curve.append(gmax)

    # Anchor at the shared starting catalyst (step 0, 37%).
    x = np.array([0] + steps)
    y = np.array([START_YIELD] + max_curve)

    fig, ax = plt.subplots(figsize=(13.5, 9.0))

    ax.plot(x, y, color=ACCENT, marker="o", markersize=9, linewidth=3,
            label=f"Ablation A — campaign {campaign_id} (best-so-far yield)", zorder=3)
    ax.axhline(89.0, color="#2ca02c", linestyle=":", linewidth=2,
               label="Best Database Hit (89.0% limit)", zorder=1)

    # Highlight the record-setting DB rediscoveries on the curve.
    hx = [h[0] for h in record_hits]
    hy = [h[1] for h in record_hits]
    ax.scatter(hx, hy, s=240, facecolor="white", edgecolor=HIT_COLOR,
               linewidth=3, zorder=4,
               label="LLM proposal = exact database catalyst")

    # Place each rediscovered structure around the plot, connected to its point.
    # (x_frac, y_frac) anchor positions in axes coords, tuned for a 3-hit story.
    box_positions = [(0.20, 0.30), (0.50, 0.30), (0.80, 0.62)]
    for i, (step, yld, smi, db_str) in enumerate(record_hits):
        pos = box_positions[i % len(box_positions)]
        img = mol_to_image(smi)
        imagebox = OffsetImage(img, zoom=0.62)
        ab = AnnotationBbox(
            imagebox, (step, yld),
            xybox=pos, xycoords="data", boxcoords="axes fraction",
            box_alignment=(0.5, 0.5), pad=0.35,
            arrowprops=dict(arrowstyle="-|>", color=HIT_COLOR, linewidth=2.2,
                            connectionstyle="arc3,rad=-0.12"),
            bboxprops=dict(edgecolor=HIT_COLOR, linewidth=2, boxstyle="round,pad=0.4"),
            zorder=5,
        )
        ax.add_artist(ab)
        ax.annotate(f"Step {step} · {db_str} (database)",
                    xy=pos, xycoords="axes fraction",
                    xytext=(0, -88), textcoords="offset points",
                    ha="center", va="top", fontsize=10.5, fontweight="bold",
                    color=HIT_COLOR, zorder=6)

    ax.set_xlabel("Optimization Step", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Maximum Discovered Yield (%)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title("Ablation A: the LLM rediscovers real database catalysts de novo",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xlim(-0.4, max(steps) + 0.6)
    ax.set_ylim(START_YIELD - 6, 95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11.5, frameon=True,
              facecolor="white", edgecolor="none")

    plt.tight_layout()
    out = out_path("ablations", f"ablation_A_knowledge_showcase_campaign_{campaign_id}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"-> Showcase figure saved to '{out}'")
    print(f"   Rediscovered {len(record_hits)} record-setting database catalysts:")
    for step, yld, smi, db_str in record_hits:
        print(f"     step {step:2d}  {db_str:>4}  {smi}")
    return out


if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    build_figure(cid)
