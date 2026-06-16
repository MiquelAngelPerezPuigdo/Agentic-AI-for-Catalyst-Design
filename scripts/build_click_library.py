"""Build the combinatorial CuAAC (click) library from the master dataset.

Idea (Self-Driving-Lab framing):
Every 1,4-disubstituted 1,2,3-triazole ligand in the master DB is the product of a
copper-catalyzed azide-alkyne cycloaddition (CuAAC) between:
  * an ALKYNE precursor   R1-C#CH   (R1 ends up on triazole C4; here R1 carries the pyridine core)
  * an AZIDE  precursor   R2-N3     (R2 ends up on triazole N1; typically a benzyl/alkyl group)

By retrosynthetically splitting each triazole into its (alkyne arm, azide arm) pair and then
taking the full cross-product of the unique arms, we recover the much larger virtual library
that the SAME building blocks could have produced. Feeding the LLM only the SHORT precursor
lists (not the full library) lets it navigate this large implied space with minimal input text.

Outputs (written to output/library/):
  click_alkynes.smi  - unique alkyne arms (dummy [1*] marks the C4 attachment)
  click_azides.smi   - unique azide arms  (dummy [2*] marks the N1 attachment)
  click_library.smi  - full enumerated, de-duplicated combinatorial library (canonical SMILES)
"""

import os
import sys

# Allow running from anywhere: ensure the repo root (parent of scripts/) is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.ligand_data import MASTER_DATASET_RAW

# 1,4-triazole core: C4 (bears alkyne arm), C5-H, N1 (bears azide arm), N2, N3
TRIAZOLE = Chem.MolFromSmarts("[c:1]1[cH:2][n:3]([*:4])[n:5][n:6]1")


def parse_master_db():
    mols = []
    for line in MASTER_DATASET_RAW.strip().split("\n"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        m = Chem.MolFromSmiles(parts[1].strip())
        if m:
            mols.append(m)
    return mols


def split_triazole(mol):
    """Return (alkyne_arm, azide_arm, core) fragments with matched dummy labels, or None."""
    matches = mol.GetSubstructMatches(TRIAZOLE)
    if not matches:
        return None
    c4, c5, n1, r_azide, n2, n3 = matches[0]
    ring = {c4, c5, n1, n2, n3}

    # exocyclic neighbour of C4 = alkyne arm attachment
    r_alkyne = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c4).GetNeighbors() if nb.GetIdx() not in ring]
    if not r_alkyne:
        return None
    r_alkyne = r_alkyne[0]

    b1 = mol.GetBondBetweenAtoms(c4, r_alkyne).GetIdx()      # C4 - R1 (alkyne)
    b2 = mol.GetBondBetweenAtoms(n1, r_azide).GetIdx()       # N1 - R2 (azide)

    # Dummy atoms carry MAP NUMBERS (1 on the C4/alkyne cut, 2 on the N1/azide cut)
    # so molzip can later pair them unambiguously.
    frag = Chem.FragmentOnBonds(mol, [b1, b2], addDummies=True,
                                dummyLabels=[(1, 1), (2, 2)])
    # FragmentOnBonds sets the dummy ISOTOPE; copy that onto the atom-map number for molzip.
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(atom.GetIsotope())
            atom.SetIsotope(0)

    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)

    alkyne_arm = azide_arm = core = None
    for pc in pieces:
        labels = {a.GetAtomMapNum() for a in pc.GetAtoms() if a.GetAtomicNum() == 0}
        if labels == {1, 2}:        # core holds both attachment points + triazole ring
            core = pc
        elif labels == {1}:
            alkyne_arm = pc
        elif labels == {2}:
            azide_arm = pc
    if alkyne_arm is None or azide_arm is None or core is None:
        return None
    return alkyne_arm, azide_arm, core


def recombine(alkyne_arm, core, azide_arm):
    """Zip an alkyne arm + core + azide arm back into a full triazole product (canonical SMILES).

    molzip joins fragments by matching dummy-atom MAP NUMBERS: map=1 dummies bond together
    (C4 reconnection) and map=2 dummies bond together (N1 reconnection).
    """
    combo = Chem.CombineMols(Chem.CombineMols(alkyne_arm, core), azide_arm)
    try:
        product = Chem.molzip(combo)
        Chem.SanitizeMol(product)
        # molzip may leave map numbers on the newly formed atoms; clear them for a clean canonical SMILES
        for atom in product.GetAtoms():
            atom.SetAtomMapNum(0)
        smi = Chem.MolToSmiles(product)
        # reject if any unjoined dummy remains
        if "*" in smi:
            return None
        return Chem.CanonSmiles(smi)
    except Exception:
        return None


def main():
    os.makedirs("output", exist_ok=True)
    mols = parse_master_db()
    print(f"Total parsed molecules in master DB: {len(mols)}")

    alkynes = {}   # canonical arm SMILES -> rdkit mol
    azides = {}
    core_ref = None
    split_ok = 0

    for m in mols:
        res = split_triazole(m)
        if res is None:
            continue
        split_ok += 1
        alkyne_arm, azide_arm, core = res
        if core_ref is None:
            core_ref = core
        alkynes.setdefault(Chem.MolToSmiles(alkyne_arm), alkyne_arm)
        azides.setdefault(Chem.MolToSmiles(azide_arm), azide_arm)

    print(f"Triazoles successfully split into precursors: {split_ok}")
    print(f"Unique ALKYNE arms (pyridine-bearing C4 synthons): {len(alkynes)}")
    print(f"Unique AZIDE  arms (N1 synthons):                  {len(azides)}")

    # Enumerate the full combinatorial cross-product
    library = set()
    for a_arm in alkynes.values():
        for z_arm in azides.values():
            prod = recombine(a_arm, core_ref, z_arm)
            if prod:
                library.add(prod)

    n_a, n_z = len(alkynes), len(azides)
    print("\n================ COMBINATORIAL CLICK LIBRARY ================")
    print(f"  Precursors fed to LLM:  {n_a} alkynes + {n_z} azides = {n_a + n_z} fragments")
    print(f"  Theoretical cross-product:  {n_a} x {n_z} = {n_a * n_z}")
    print(f"  Unique valid products enumerated: {len(library)}")
    print(f"  Compression: {n_a + n_z} input fragments imply a {len(library)}-molecule search space "
          f"({len(library) / (n_a + n_z):.1f}x leverage)")
    print("=============================================================")

    from src.paths import out_path
    p_alk = out_path("library", "click_alkynes.smi")
    p_azi = out_path("library", "click_azides.smi")
    p_lib = out_path("library", "click_library.smi")
    with open(p_alk, "w") as f:
        f.write("\n".join(sorted(alkynes.keys())))
    with open(p_azi, "w") as f:
        f.write("\n".join(sorted(azides.keys())))
    with open(p_lib, "w") as f:
        f.write("\n".join(sorted(library)))
    print(f"\n[+] Saved: {p_alk}, {p_azi}, {p_lib}")


if __name__ == "__main__":
    main()
