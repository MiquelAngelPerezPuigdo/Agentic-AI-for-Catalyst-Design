"""UMAP of MBH catalysts (score > threshold) for the Gemini campaigns, grouped by
Murcko scaffold (nitrogen preserved & highlighted), distinguishing the neutrality
filter (baseline = charge allowed, neutral = charge forbidden).

Auto-discovers Gemini ``run_*`` dirs and classifies them by name ('_neutral_' ->
neutral, else baseline). Outputs a PNG + CSV under output/saturn_mbh/.

Run:  conda run -n ppchem python scripts/umap_mbh_scaffolds.py
"""
import glob
import os

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
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
cmap = plt.get_cmap("tab10")
scaf_color = {s: cmap(i) for i, s in enumerate(top_scaffolds)}

def color_for(s):
    return scaf_color.get(s, (0.6, 0.6, 0.6, 1.0))

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0])
ax = fig.add_subplot(gs[0, 0])

markers = {"baseline": "o", "neutral": "^"}
for cls in ["baseline", "neutral"]:
    sub = data[data.cls == cls]
    ax.scatter(sub.x, sub.y, s=70, alpha=0.85,
               c=[color_for(s) for s in sub.scaffold],
               marker=markers[cls], edgecolors="black", linewidths=0.4)

ax.set_title(f"UMAP of MBH catalysts (score>{SCORE_MIN:.0f}) — colored by Murcko scaffold (N kept)\n"
             "circle = baseline filter,  triangle = neutral filter", fontsize=12)
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
ax.grid(alpha=0.2)

legend_items = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=scaf_color[s], markeredgecolor="black",
                       markersize=10, label=f"scaffold #{i+1}  (n={counts[s]})")
                for i, s in enumerate(top_scaffolds)]
legend_items.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.6, 0.6, 0.6),
                           markeredgecolor="black", markersize=10,
                           label=f"other  (n={len(data) - counts[top_scaffolds].sum()})"))
legend_items += [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=10, label="baseline (circle)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=10, label="neutral (triangle)"),
]
ax.legend(handles=legend_items, fontsize=9, loc="best", framealpha=0.9)

# Draw the scaffolds WITH the nitrogen highlighted.
mols = [Chem.MolFromSmiles(s) for s in top_scaffolds]
highlights = [[a.GetIdx() for a in m.GetAtoms() if m and a.GetSymbol() == "N"] for m in mols]
legends = [f"#{i+1}  n={counts[s]}" for i, s in enumerate(top_scaffolds)]
grid = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(220, 180),
                            legends=legends, highlightAtomLists=highlights)
gax = fig.add_subplot(gs[0, 1])
gax.imshow(grid)
gax.axis("off")
gax.set_title("Top Murcko scaffolds (N highlighted)", fontsize=12)

fig.tight_layout()
out = os.path.join(BASE, "umap_gemini_scaffold_N_gt50.png")
fig.savefig(out, dpi=150)
print("Saved:", out)

print("\nTop N-containing Murcko scaffolds:")
for i, s in enumerate(top_scaffolds):
    sub = data[data.scaffold == s]
    nb = (sub.cls == "baseline").sum(); nn = (sub.cls == "neutral").sum()
    print(f"  #{i+1}: n={counts[s]:3d}  (baseline {nb}, neutral {nn})  meanScore={sub.score.mean():.1f}  {s}")
