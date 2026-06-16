"""Showcase figure for the thesis: the five Ablation-A campaigns drawn together with the
real database catalysts the LLM rediscovered along the way, connected to the step at which
they were first proposed.

Narrative: under the unbiased baseline (Ablation A) the agent repeatedly proposes *real*
literature catalysts spanning the whole yield range, progressively refining one pharmacophore
(1,2,3-triazole + hydroxypyridine directing group, then CF3-loaded arms). One representative
campaign supplies the annotated molecules so the climb from a modest known catalyst (37%) up
to the database champion (89%) is told through actual structures.

Run:
    conda run -n ppchem python scripts/plot_ablationA_showcase.py
Output:
    output/ablations/ablationA_knowledge_showcase.png
"""

import os
import sys
import glob
import json

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

from src.ligand_data import MASTER_DATASET_RAW
from src.paths import out_path

START_YIELD = 37.0

# Curated real DB catalysts the LLM rediscovered, picked across DIFFERENT campaigns to span the
# whole yield range. Each entry is (campaign_id, canonical_smiles); the step and yield are read
# back from that campaign's saved proposals so the star lands on the true proposal point.
ANNOTATED_MOLECULES = [
    (2, "CC(C)(c1cn(Cc2ccccc2)nn1)c1cccc(O)n1"),                                   # ~21%
    (1, "Oc1nc(-c2cn(Cc3ccccc3)nn2)ccc1Cl"),                                       # ~52%
    (1, "COc1cc(Cl)c(-c2cn(Cc3ccccc3)nn2)nc1O"),                                   # ~74%
    (5, "Oc1ccc(C(F)(F)F)c(-c2cn(Cc3ccccc3)nn2)n1"),                               # ~80%
    (1, "COc1cc(-c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c(-c2cn(Cc3ccc(C(F)(F)F)cc3)nn2)nc1O"),  # 89% champion
]


def build_db():
    """Canonical SMILES -> reported yield string from the master dataset."""
    db = {}
    for line in MASTER_DATASET_RAW.strip().split("\n"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        m = Chem.MolFromSmiles(parts[1].strip())
        if m:
            db[Chem.MolToSmiles(m)] = parts[0]
    return db


def campaign_trajectory(logs):
    """Per-step running-max yield trajectory, anchored at (step 0, 37%)."""
    cur = 0.0
    traj = []
    for s in logs:
        cur = max(cur, max(p["yield"] for p in s["proposals"]))
        traj.append(cur)
    return np.concatenate(([START_YIELD], traj))


def locate_proposal(logs, smiles):
    """Find the (step, yield) at which a given canonical SMILES was proposed in a campaign."""
    for s in logs:
        for pr in s["proposals"]:
            if pr["canonical_smiles"] == smiles:
                return s["step"], pr["yield"]
    return None


def mol_image(smiles, bond_px=26, canvas=1000, margin=12):
    """Render a molecule at a FIXED bond length and autocrop to its bounding box.

    Because every molecule is drawn with the same pixels-per-bond, the cropped image size is
    proportional to the molecule's real extent: bigger molecules produce bigger images. Using a
    single OffsetImage zoom for all panels then preserves that physical scale on the figure.
    """
    import io
    from PIL import Image
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(canvas, canvas)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.fixedBondLength = bond_px          # identical bond length for every molecule
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    img = Image.open(io.BytesIO(d.GetDrawingText())).convert("RGBA")
    bbox = img.split()[-1].getbbox()        # crop to non-transparent content
    if bbox:
        l, t, r, b = bbox
        l, t = max(0, l - margin), max(0, t - margin)
        r, b = min(img.width, r + margin), min(img.height, b + margin)
        img = img.crop((l, t, r, b))
    return np.asarray(img)


def main():
    files = sorted(glob.glob(out_path("campaign_details", "campaign_details_A_campaign_*.json")))
    if not files:
        raise SystemExit("[!] No A campaign_details found. Run the A campaign with --save-details first.")

    db = build_db()

    # Load every campaign, keyed by its campaign id.
    campaigns = {}
    for f in files:
        cid = int(f.split("_campaign_")[-1].split(".")[0])
        campaigns[cid] = json.load(open(f))

    # Build trajectories on a common step axis (pad early-stopped runs with their final value).
    max_steps = max(len(logs) for logs in campaigns.values())
    x = np.arange(0, max_steps + 1)
    trajs = {}
    for cid, logs in campaigns.items():
        t = campaign_trajectory(logs)
        if len(t) < len(x):
            t = np.concatenate([t, np.full(len(x) - len(t), t[-1])])
        trajs[cid] = t
    mean_traj = np.mean(np.vstack(list(trajs.values())), axis=0)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Five individual campaigns as thin dashed lines.
    for cid in sorted(trajs):
        ax.plot(x, trajs[cid], color="#9bb8f0", linestyle="--", linewidth=1.4,
                marker="o", markersize=4, alpha=0.9, zorder=2,
                label="Individual campaigns" if cid == min(trajs) else None)

    # Bold mean trajectory.
    ax.plot(x, mean_traj, color="#1a4fb0", linewidth=3.4, marker="o", markersize=8,
            zorder=4, label="Mean of 5 campaigns")

    ax.axhline(89.0, color="#2ca02c", linestyle=":", linewidth=2,
               label="Database champion (89.0%)")
    ax.axhline(START_YIELD, color="#999999", linestyle="--", linewidth=1.2,
               label="Starting catalyst (37.0%)")

    # Locate each curated molecule at the true (step, yield) where its campaign proposed it.
    events = []
    for cid, smi in ANNOTATED_MOLECULES:
        if cid not in campaigns:
            continue
        loc = locate_proposal(campaigns[cid], smi)
        if loc is None:
            print(f"[!] Molecule not found in campaign {cid}: {smi}")
            continue
        step, yld = loc
        events.append((cid, step, yld, smi))
    events.sort(key=lambda e: e[1])   # order the molecule row by step (keeps connectors from crossing)

    # All molecules live in a reserved whitespace band BELOW the rising curves, laid out in a tidy
    # evenly-spaced row so nothing covers the data. A single shared zoom keeps the
    # fixed-bond-length images at true relative scale (bigger molecule -> bigger panel).
    n = len(events)
    row_y = 0.16                      # axes-fraction height of the molecule row (lower whitespace)
    xfracs = np.linspace(0.11, 0.90, n)
    for (cid, step, yld, smi), fx in zip(events, xfracs):
        img = mol_image(smi)
        oi = OffsetImage(img, zoom=0.34)
        ab = AnnotationBbox(
            oi, (step, yld),
            xybox=(fx, row_y), xycoords="data", boxcoords="axes fraction",
            box_alignment=(0.5, 0.5), pad=0.3,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=1.1,
                            connectionstyle="arc3,rad=0.0"),
            bboxprops=dict(edgecolor="#D97757", linewidth=1.6, boxstyle="round,pad=0.3",
                           facecolor="white"),
            zorder=6,
        )
        ax.add_artist(ab)
        ax.scatter([step], [yld], s=250, marker="*", color="#D97757",
                   edgecolor="black", linewidth=0.8, zorder=7)
        ax.annotate(f"{yld:.0f}%  (C{cid})", (step, yld), textcoords="offset points",
                    xytext=(7, 8), fontsize=10.5, fontweight="bold",
                    color="#222222", zorder=8)

    ax.set_xlabel("Optimization Step", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Maximum Discovered Yield (%)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title("Ablation A: the LLM rediscovers real literature catalysts as it climbs",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlim(-0.4, max_steps + 0.6)
    ax.set_ylim(20, 90)
    ax.set_xticks(range(0, max_steps + 1))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", bbox_to_anchor=(0.30, 0.99), fontsize=11,
              frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    output_path = out_path("ablations", "ablationA_knowledge_showcase.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Showcase figure saved to '{output_path}'.")
    print("   Annotated DB rediscoveries (cross-campaign): "
          f"{[(f'C{cid}', s, f'{y:.0f}%') for cid, s, y, _ in events]}")


if __name__ == "__main__":
    main()
