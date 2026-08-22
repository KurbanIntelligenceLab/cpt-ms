"""Energy screening of generated structures with an ML potential."""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from cpt.common.artifacts import emit

SCRIPT = "src/cpt/crystal/energy_screen.py"
MIN_OCCURRENCES = 10


def score_structures(cifs, labels=None, cache: Path | None = None):
    """Total energy per atom from CHGNet, with composition and optional DFT label."""
    if cache and cache.exists():
        return pickle.load(open(cache, "rb"))
    from chgnet.model import CHGNet
    from pymatgen.core import Structure
    model = CHGNet.load()
    recs = []
    for i, cif in enumerate(cifs):
        try:
            st = Structure.from_str(cif, fmt="cif")
            e_per_atom = float(model.predict_structure(st)["e"])
            comp = {str(k): float(v) for k, v in st.composition.get_el_amt_dict().items()}
            recs.append({
                "n_atoms": len(st), "e_total": e_per_atom * len(st), "comp": comp,
                "label_ef": (float(labels[i]) if labels is not None else None),
            })
        except Exception:
            continue
    if cache:
        pickle.dump(recs, open(cache, "wb"))
    return recs


def fit_references(recs, min_occ: int = MIN_OCCURRENCES):
    """Least-squares elemental references from labelled structures."""
    labelled = [r for r in recs if r["label_ef"] is not None]
    occ = Counter(e for r in labelled for e in r["comp"])
    keep = {e for e, k in occ.items() if k >= min_occ}
    use = [r for r in labelled if set(r["comp"]) <= keep]
    els = sorted(keep)
    if not use or not els:
        raise ValueError(f"no structures survive min_occ={min_occ}")
    A = np.zeros((len(use), len(els)))
    y = np.zeros(len(use))
    for i, r in enumerate(use):
        for e, n in r["comp"].items():
            A[i, els.index(e)] = n
        y[i] = r["e_total"] - r["n_atoms"] * r["label_ef"]
    mu, *_ = np.linalg.lstsq(A, y, rcond=None)
    rank = int(np.linalg.matrix_rank(A))
    return dict(zip(els, mu.tolist(), strict=False)), keep, rank, len(use)


def formation_energy(rec, refs) -> float | None:
    """Formation energy per atom, or None if any element lacks a reference."""
    if not set(rec["comp"]) <= set(refs):
        return None
    ref = sum(refs[e] * n for e, n in rec["comp"].items())
    return (rec["e_total"] - ref) / rec["n_atoms"]


def main(n_calib: int = 3000, seed: int = 0, smoke: bool = False) -> Path:
    import pandas as pd
    n_calib = 300 if smoke else n_calib
    d = pd.read_csv("data/mp20_train.csv").sample(n_calib, random_state=seed)
    cache = Path(f"/tmp/chgnet_scores_{n_calib}_{seed}.pkl")
    recs = score_structures(list(d.cif), list(d.formation_energy_per_atom), cache=cache)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))
    cut = len(recs) // 2
    fit_recs = [recs[i] for i in idx[:cut]]
    test_recs = [recs[i] for i in idx[cut:]]

    refs, _keep, rank, n_used = fit_references(fit_recs)

    pred, true = [], []
    excluded = 0
    for r in test_recs:
        ef = formation_energy(r, refs)
        if ef is None:
            excluded += 1
            continue
        pred.append(ef)
        true.append(r["label_ef"])
    pred, true = np.array(pred), np.array(true)
    err = pred - true

    payload = {
        "surrogate": "CHGNet v0.3.0",
        "is_dft": False,
        "disclaimer": (
            "ML-potential screen, not DFT. Every downstream stability number derived "
            "from this calibration must be labelled an ML-potential screen. A working "
            "DFT engine (GPAW 25.7.0, verified running on the compute box) IS "
            "available; the surrogate is used because MP-20 already carries DFT labels "
            "for these reference structures, not because DFT is unavailable. Neither "
            "DFT nor this surrogate applies to GENERATED samples, whose serialization "
            "carries no atomic positions."
        ),
        "calibration": {
            "n_scored": len(recs), "n_fit_split": len(fit_recs),
            "n_test_split": len(test_recs), "n_used_in_fit": n_used,
            "min_element_occurrences": MIN_OCCURRENCES,
            "n_element_references": len(refs), "design_matrix_rank": rank,
            "rank_deficient": rank < len(refs),
        },
        "held_out_agreement": {
            "n_evaluated": len(pred),
            "n_excluded_rare_elements": int(excluded),
            "mae_ev_per_atom": float(np.abs(err).mean()) if len(err) else None,
            "rmse_ev_per_atom": float(np.sqrt((err ** 2).mean())) if len(err) else None,
            "bias_ev_per_atom": float(err.mean()) if len(err) else None,
            "pearson_r": (float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 2 else None),
            "label_spread_ev_per_atom": float(true.std()) if len(true) else None,
        },
        "element_references_ev": refs,
    }
    return emit(
        "energy_calibration" + ("_smoke" if smoke else ""),
        payload,
        config={"n_calib": n_calib, "seed": seed, "min_occ": MIN_OCCURRENCES,
                "smoke": smoke},
        seeds=[seed], script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(n_calib=a.n_calib, seed=a.seed, smoke=a.smoke))
