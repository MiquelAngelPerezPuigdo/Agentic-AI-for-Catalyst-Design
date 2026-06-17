"""Click (CuAAC) combinatorial library: building blocks + pair -> product recombination.

This module exposes the alkyne / azide building blocks recovered from the master dataset
(see build_click_library.py for the enumeration logic) so a prospective campaign can feed the
LLM only the short precursor lists and accept compact (alkyne_id, azide_id) proposals.

Public API:
    get_library()           -> dict with ordered alkynes, azides, reagent SMILES, and product map
"""

import os
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.ligand_data import MASTER_DATASET_RAW

# 1,4-triazole core: C4 (alkyne arm), C5-H, N1 (azide arm), N2, N3
TRIAZOLE = Chem.MolFromSmarts("[c:1]1[cH:2][n:3]([*:4])[n:5][n:6]1")

_CACHE = None


def _parse_db():
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


def _split(mol):
    matches = mol.GetSubstructMatches(TRIAZOLE)
    if not matches:
        return None
    c4, c5, n1, r_azide, n2, n3 = matches[0]
    ring = {c4, c5, n1, n2, n3}
    r_alkyne = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c4).GetNeighbors() if nb.GetIdx() not in ring]
    if not r_alkyne:
        return None
    r_alkyne = r_alkyne[0]
    b1 = mol.GetBondBetweenAtoms(c4, r_alkyne).GetIdx()
    b2 = mol.GetBondBetweenAtoms(n1, r_azide).GetIdx()
    frag = Chem.FragmentOnBonds(mol, [b1, b2], addDummies=True, dummyLabels=[(1, 1), (2, 2)])
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(atom.GetIsotope())
            atom.SetIsotope(0)
    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    alkyne_arm = azide_arm = core = None
    for pc in pieces:
        labels = {a.GetAtomMapNum() for a in pc.GetAtoms() if a.GetAtomicNum() == 0}
        if labels == {1, 2}:
            core = pc
        elif labels == {1}:
            alkyne_arm = pc
        elif labels == {2}:
            azide_arm = pc
    if alkyne_arm is None or azide_arm is None or core is None:
        return None
    return alkyne_arm, azide_arm, core


def _arm_to_reagent(arm_mol, kind):
    """Convert a dummy-labelled arm fragment to the real reagent SMILES.

    alkyne arm  R-[1*]  -> terminal alkyne  R-C#CH
    azide  arm  R-[2*]  -> organic azide    R-N=[N+]=[N-]
    """
    rw = Chem.RWMol(arm_mol)
    dummy = next(a for a in rw.GetAtoms() if a.GetAtomicNum() == 0)
    nbr = dummy.GetNeighbors()[0].GetIdx()
    rw.RemoveAtom(dummy.GetIdx())
    # re-fetch neighbour index after removal
    nbr = next(a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() == nbr) if nbr < rw.GetNumAtoms() else None
    # simpler: rebuild by attaching the proper group via SMILES surgery
    if kind == "alkyne":
        cap = Chem.MolFromSmiles("C#C")  # terminal alkyne stub; attach R to first C
    else:
        cap = Chem.MolFromSmiles("N=[N+]=[N-]")
    # Use molzip-style join: put a map-1 dummy back on the arm and on the cap, then zip
    arm = Chem.RWMol(arm_mol)
    for a in arm.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomMapNum(1)
            a.SetIsotope(0)
    if kind == "alkyne":
        capm = Chem.MolFromSmiles("[*:1]C#C")
    else:
        capm = Chem.MolFromSmiles("[*:1]N=[N+]=[N-]")
    combo = Chem.CombineMols(arm.GetMol(), capm)
    try:
        prod = Chem.molzip(combo)
        Chem.SanitizeMol(prod)
        for a in prod.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.CanonSmiles(Chem.MolToSmiles(prod))
    except Exception:
        return Chem.MolToSmiles(arm_mol)


def _build():
    mols = _parse_db()
    alkynes, azides = {}, {}
    core_ref = None
    for m in mols:
        res = _split(m)
        if res is None:
            continue
        a_arm, z_arm, core = res
        if core_ref is None:
            core_ref = core
        alkynes.setdefault(Chem.MolToSmiles(a_arm), a_arm)
        azides.setdefault(Chem.MolToSmiles(z_arm), z_arm)

    alkyne_arms = [alkynes[k] for k in sorted(alkynes)]
    azide_arms = [azides[k] for k in sorted(azides)]

    alkyne_reagents = [_arm_to_reagent(a, "alkyne") for a in alkyne_arms]
    azide_reagents = [_arm_to_reagent(z, "azide") for z in azide_arms]

    # Pre-enumerate all products keyed by (a_idx, z_idx)
    product_map = {}
    for ai, a_arm in enumerate(alkyne_arms):
        for zi, z_arm in enumerate(azide_arms):
            combo = Chem.CombineMols(Chem.CombineMols(a_arm, core_ref), z_arm)
            try:
                prod = Chem.molzip(combo)
                Chem.SanitizeMol(prod)
                for at in prod.GetAtoms():
                    at.SetAtomMapNum(0)
                smi = Chem.MolToSmiles(prod)
                if "*" not in smi:
                    product_map[(ai, zi)] = Chem.CanonSmiles(smi)
            except Exception:
                pass

    return {
        "alkyne_arms": alkyne_arms,
        "azide_arms": azide_arms,
        "alkyne_reagents": alkyne_reagents,
        "azide_reagents": azide_reagents,
        "core": core_ref,
        "product_map": product_map,
        "n_alkynes": len(alkyne_arms),
        "n_azides": len(azide_arms),
    }


def get_library():
    global _CACHE
    if _CACHE is None:
        _CACHE = _build()
    return _CACHE


def format_building_blocks():
    """Return a compact prompt-ready listing of the alkyne and azide building blocks."""
    lib = get_library()
    lines = ["ALKYNE building blocks (terminal alkynes, R-C#CH):"]
    for i, smi in enumerate(lib["alkyne_reagents"]):
        lines.append(f"  A{i}: {smi}")
    lines.append("\nAZIDE building blocks (organic azides, R-N3):")
    for i, smi in enumerate(lib["azide_reagents"]):
        lines.append(f"  Z{i}: {smi}")
    return "\n".join(lines)


if __name__ == "__main__":
    lib = get_library()
    print(f"Alkynes: {lib['n_alkynes']}  Azides: {lib['n_azides']}  Products: {len(lib['product_map'])}")
    print()
    print(format_building_blocks())
