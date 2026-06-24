"""UMAP of MBH catalysts (score > threshold) for the Gemini campaigns, grouped by
Murcko scaffold (nitrogen preserved & highlighted), distinguishing the neutrality
filter (baseline = charge allowed, neutral = charge forbidden).

Auto-discovers Gemini ``run_*`` dirs and classifies them by name ('_neutral_' ->
neutral, else baseline). Outputs a PNG + CSV under output/saturn_mbh/.

Run:  conda run -n ppchem python scripts/umap_mbh_scaffolds.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # so 'src' is importable when run as a script
from src.report import set_publication_style

BASE = os.path.join(REPO, "output", "saturn_mbh")
SCORE_COLS = ("agentic_mbh_catalyst_score_raw_values", "MBH_catalyst_score_raw_values")
SCORE_MIN = 50.0


def discover_gemini_campaigns():
    """Yield (run_dir_name, class) for every Gemini run; '_neutral_' -> neutral."""
    for path in sorted(glob.glob(os.path.join(BASE, "run_*gemini*"))):
        name = os.path.basename(path)
        if not os.path.isfile(os.path.join(path, "oracle_history.csv")):
            continue
        yield name, ("neutral" if "_neutral_" in name else "baseline")


def murcko_real(smi):
    """Real Murcko scaffold (ring system WITH heteroatoms, so the nitrogen position
    is preserved). Acyclic -> the molecule itself."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf if scaf.GetNumAtoms() else m)
    except Exception:
        return Chem.MolToSmiles(m)


rows = []
for d, cls in discover_gemini_campaigns():
    df = pd.read_csv(os.path.join(BASE, d, "oracle_history.csv"))
    score_col = next(c for c in SCORE_COLS if c in df.columns)
    df = df.dropna(subset=[score_col, "smiles"])
    df = df[df[score_col] > SCORE_MIN]
    for _, r in df.iterrows():
        rows.append((r["smiles"], float(r[score_col]), cls, d))

data = pd.DataFrame(rows, columns=["smiles", "score", "cls", "campaign"])
data["scaffold"] = data.smiles.apply(murcko_real)
data = data.dropna(subset=["scaffold"]).reset_index(drop=True)
print(f"Molecules score>{SCORE_MIN:.0f}: {len(data)} | unique N-containing Murcko scaffolds: {data.scaffold.nunique()}")

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps, keep = [], []
for i, smi in enumerate(data.smiles):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    fps.append(np.array(mfpgen.GetFingerprint(m)))
    keep.append(i)
data = data.iloc[keep].reset_index(drop=True)
X = np.array(fps)

emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard",
                random_state=42).fit_transform(X)
data["x"], data["y"] = emb[:, 0], emb[:, 1]

counts = data.scaffold.value_counts()
TOPN = 8
top_scaffolds = list(counts.index[:TOPN])
# Colorblind-safe qualitative palette (matches the repo's publication style).
GREY = (0.6, 0.6, 0.6, 1.0)
palette = sns.color_palette("colorblind", TOPN)
scaf_color = {s: palette[i] for i, s in enumerate(top_scaffolds)}

def color_for(s):
    return scaf_color.get(s, GREY)

# Apply the shared, thesis-quality theme used by all other figures in the repo.
set_publication_style()
fig, ax = plt.subplots(figsize=(12, 11))

markers = {"baseline": "o", "neutral": "^"}
for cls in ["baseline", "neutral"]:
    sub = data[data.cls == cls]
    ax.scatter(sub.x, sub.y, s=80, alpha=0.85,
               c=[color_for(s) for s in sub.scaffold],
               marker=markers[cls], edgecolors="black", linewidths=0.4)

# No title (consistent with other repo figures); labels and axes are clean
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

# Embed scaffold molecule drawings directly inside the plot's open spaces
# with lines pointing to their cluster centroids.
OFFSETS = [
    (-100, -50), # #1 (pyrrolizidine bottom-left)
    (-40, 70),   # #2 (indolizidine top-left)
    (-100, 30),  # #3 (cyclobutane fused top-left)
    (50, 80),    # #4 (quinuclidine top-right)
    (90, -45),   # #5 (fused azabicyclic center-right)
    (95, 45),    # #6 (fused azabicyclic variant)
    (95, 0),     # #7 (bridged azabicyclic far-right)
    (-100, -10), # #8 (fused azabicyclic left)
]

for i, s in enumerate(top_scaffolds):
    sub_scaf = data[data.scaffold == s]
    centroid_x = sub_scaf["x"].mean()
    centroid_y = sub_scaf["y"].mean()
    
    # Draw scaffold and highlight the bridgehead Nitrogen atom
    m = Chem.MolFromSmiles(s)
    highlight = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N"]
    img = Draw.MolToImage(m, size=(100, 80), highlightAtoms=highlight, highlightColor=(0.85, 0.15, 0.15, 0.35))
    
    # Create matplotlib imagebox
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    imagebox = OffsetImage(img, zoom=0.75)
    
    # Place on plot with an arrow pointing to the cluster centroid
    ab = AnnotationBbox(
        imagebox,
        (centroid_x, centroid_y),
        xybox=OFFSETS[i],
        xycoords="data",
        boxcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2, alpha=0.7),
        bboxprops=dict(edgecolor=scaf_color[s], lw=1.5, facecolor="white", alpha=0.9, boxstyle="round,pad=0.15")
    )
    ax.add_artist(ab)
    
    # Label the box with its matching scaffold rank
    # Position text slightly above/left of the image box
    tx = centroid_x
    ty = centroid_y
    ax.annotate(f"#{i+1}", xy=(tx, ty), xytext=(OFFSETS[i][0] - 40, OFFSETS[i][1] + 35),
                textcoords="offset points", fontsize=10, weight="bold", color="#2c3e50")

# Expand axis limits to comfortably fit the annotated boxes
ax.set_xlim(data.x.min() - 4.5, data.x.max() + 4.5)
ax.set_ylim(data.y.min() - 3.5, data.y.max() + 3.5)

legend_items = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=scaf_color[s], markeredgecolor="black",
                       markersize=10, label=f"scaffold #{i+1}  (n={counts[s]})")
                for i, s in enumerate(top_scaffolds)]
legend_items.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
                           markeredgecolor="black", markersize=10,
                           label=f"other  (n={len(data) - counts[top_scaffolds].sum()})"))
legend_items += [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=10, label="baseline (circle)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=10, label="neutral (triangle)"),
]
ax.legend(handles=legend_items, loc="upper left", framealpha=0.9)

fig.tight_layout()
out = os.path.join(BASE, "umap_gemini_scaffold_N_gt50.png")
fig.savefig(out)
print("Saved:", out)

print("\nTop N-containing Murcko scaffolds:")
for i, s in enumerate(top_scaffolds):
    sub = data[data.scaffold == s]
    nb = (sub.cls == "baseline").sum(); nn = (sub.cls == "neutral").sum()
    print(f"  #{i+1}: n={counts[s]:3d}  (baseline {nb}, neutral {nn})  meanScore={sub.score.mean():.1f}  {s}")
