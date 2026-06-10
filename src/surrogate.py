import numpy as np
from src.ligand_data import MASTER_DATASET_RAW
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import Descriptors  
    from rdkit.Chem import BRICS
    from rdkit import RDLogger
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    RDLogger.DisableLog('rdApp.*')
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

class HybridSurrogateModel:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ridge = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        X_scaled = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y)
        
    def predict(self, X_query):
        rf_pred = self.rf.predict(X_query)[0]
        X_scaled = self.scaler.transform(X_query)
        ridge_pred = self.ridge.predict(X_scaled)[0]
        final_blend = (0.6 * rf_pred) + (0.4 * ridge_pred)
        return min(max(final_blend, 0.0), 98.0)

import threading

# Singleton variables to prevent retraining the model for every parallel thread
_SURROGATE_LOCK = threading.Lock()
_SURROGATE_INSTANCE = None
_FP_GENERATOR = None
_DATASET_DICT = None

def init_surrogate():
    global _SURROGATE_INSTANCE, _FP_GENERATOR, _DATASET_DICT
    with _SURROGATE_LOCK:
        if _SURROGATE_INSTANCE is not None:
            return _SURROGATE_INSTANCE, _FP_GENERATOR, _DATASET_DICT

        if not HAS_ML_LIBS:
            raise ImportError("RDKit or scikit-learn missing. Check env.yml.")

        _FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        _DATASET_DICT = {}
        X_train_fps, y_train_yields = [], []

    print("-> Training Ensemble Descriptor-Hybridized Extrapolative Surrogate Regressor...")
    for line in MASTER_DATASET_RAW.strip().split("\n"):
        if not line: continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            yield_val, smiles_val = parts[0].strip(), parts[1].strip()
            try:
                mol = Chem.MolFromSmiles(smiles_val)
                if mol:
                    canonical_key = Chem.MolToSmiles(mol, canonical=True)
                    _DATASET_DICT[canonical_key] = yield_val
                    fp = _FP_GENERATOR.GetFingerprint(mol)
                    num_yield = float(yield_val.replace("%", ""))
                    tpsa, logp = Descriptors.TPSA(mol), Descriptors.MolLogP(mol)
                    X_train_fps.append(np.append(np.array(fp), [tpsa, logp]))
                    y_train_yields.append(num_yield)
            except Exception: pass

    _SURROGATE_INSTANCE = HybridSurrogateModel()
    _SURROGATE_INSTANCE.fit(np.array(X_train_fps), np.array(y_train_yields))
    print("-> Extrapolative Hybrid Model calibrated and active across structural boundaries!\n")
    return _SURROGATE_INSTANCE, _FP_GENERATOR, _DATASET_DICT

def score_ligand(smiles, surrogate_model, fp_generator, dataset_dict, history_records):
    """Evaluates a proposed ligand via exact-match, or the SMARTS/BRICS surrogate logic."""
    canonical_lookup_key = None
    try:
        test_mol = Chem.MolFromSmiles(smiles)
        if test_mol: canonical_lookup_key = Chem.MolToSmiles(test_mol, canonical=True)
    except Exception: test_mol = None

    # 1. Check exact match
    if canonical_lookup_key and canonical_lookup_key in dataset_dict:
        return float(dataset_dict[canonical_lookup_key].replace("%", "")), canonical_lookup_key

    # 2. Check competence & Predict via ML
    chosen_yield_numeric = 0.0
    if surrogate_model and fp_generator and test_mol:
        try:
            # SMARTS filter for required structural features
            is_incompetent_core = True  
            cmd_t1 = Chem.MolFromSmarts("[O,O-]-[c,C]1:[n,N]:[c,C]:[c,C]:[c,C]:[c,C]1")
            cmd_t2 = Chem.MolFromSmarts("[O,O-]=[c,C]1:[n,N]:[c,C]:[c,C]:[c,C]:[c,C]1")
            
            if test_mol.HasSubstructMatch(cmd_t1) or test_mol.HasSubstructMatch(cmd_t2):
                ring_info = test_mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if len(ring) == 6 and any(test_mol.GetAtomWithIdx(idx).GetSymbol() == 'N' for idx in ring):
                        if not any(sum(1 for other in ring_info.AtomRings() if idx in other) > 1 for idx in ring):
                            is_incompetent_core = False
                            break  
            
            if not is_incompetent_core:
                working_mol = Chem.Mol(test_mol)
                Chem.SanitizeMol(working_mol)
                Chem.Kekulize(working_mol, clearAromaticFlags=True)
                
                # BRICS Fragmentation analysis
                frag_smiles_list = Chem.MolToSmiles(BRICS.BreakBonds(working_mol)).split('.')
                fragment_scores = []
                for frag in frag_smiles_list:
                    frag_mol = Chem.MolFromSmiles(frag)
                    if not frag_mol: continue
                    clean_mol = Chem.DeleteSubstructs(frag_mol, Chem.MolFromSmarts('[#0]'))
                    clean_frag = Chem.MolToSmiles(clean_mol, canonical=True)
                    if len(clean_frag) < 3: continue
                    
                    hist_yields = [float(h["yield"].replace("%", "")) for h in history_records if clean_frag in h["smiles"]]
                    if hist_yields: fragment_scores.append(np.mean(hist_yields))
                
                # Predict Hybrid Value
                test_fp = fp_generator.GetFingerprint(test_mol)
                hybrid_query = np.append(np.array(test_fp), [Descriptors.TPSA(test_mol), Descriptors.MolLogP(test_mol)])
                raw_ml_val = surrogate_model.predict([hybrid_query])
                
                if len(fragment_scores) >= 2 and all(score >= 40.0 for score in fragment_scores):
                    floor_premium = min(max(fragment_scores) + 5.0, 95.0)
                    chosen_yield_numeric = max(raw_ml_val, floor_premium)
                else:
                    chosen_yield_numeric = raw_ml_val
        except Exception:
            # Fallback simple prediction
            try:
                test_fp = fp_generator.GetFingerprint(test_mol)
                hybrid_query = np.append(np.array(test_fp), [Descriptors.TPSA(test_mol), Descriptors.MolLogP(test_mol)])
                chosen_yield_numeric = surrogate_model.predict([hybrid_query])
            except Exception: pass

    return chosen_yield_numeric, canonical_lookup_key or smiles