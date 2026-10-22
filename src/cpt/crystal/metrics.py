from __future__ import annotations

from collections import Counter

import numpy as np

from cpt.crystal.mp20 import VALENCE


def is_structurally_valid(toks: list[str]) -> bool:
    """One well-formed composition and cell, with a count for every element named."""
    return (toks.count("<COMP>") == 1 and toks.count("<END_COMP>") == 1
            and toks.count("<CELL>") == 1 and toks.count("<END_CELL>") == 1
            and sum(t.startswith("len:") for t in toks) == 3
            and sum(t.startswith("ang:") for t in toks) == 3
            and sum(t.startswith("el:") for t in toks)
            == sum(t.startswith("cnt:") for t in toks)
            and sum(t.startswith("el:") for t in toks) >= 1)


def structural_metrics(seqs, V, gm, cfg, train_formulas=None,
                       full_element_freq=None) -> dict:
    """Validity, density, and DISTRIBUTIONAL COVERAGE of generated sequences."""
    dens, vdens, valid, complete = [], [], 0, 0
    el_counter: Counter = Counter()
    formulas: list[tuple] = []
    for s in seqs:
        toks = [V.tokens[i] for i in s]
        ok = is_structurally_valid(toks)
        valid += ok
        if ok:
            f = {}
            els = [t[3:] for t in toks if t.startswith("el:")]
            cnts = [int(t[4:]) for t in toks if t.startswith("cnt:")]
            for e, k in zip(els, cnts, strict=False):
                f[e] = f.get(e, 0) + k
                el_counter[e] += k
            formulas.append(tuple(sorted(f.items())))
        d = V.decode_density(s) if ok else None
        if d is not None and np.isfinite(d) and 0 < d < 100:
            complete += 1
            dens.append(d)
            vdv = V.decode_weighted_density(s, VALENCE)
            if vdv is not None and np.isfinite(vdv):
                vdens.append(vdv)
    d = np.asarray(dens)
    vd = np.asarray(vdens)
    return {
        "n_samples": len(seqs),
        "structural_validity_rate": valid / len(seqs),
        "density_computable_rate": complete / len(seqs),
        "mean_density": float(d.mean()) if len(d) else None,
        "median_density": float(np.median(d)) if len(d) else None,
        "density_sd": float(d.std()) if len(d) else None,
        "density_error_vs_target": (float(d.mean() - cfg.density_target)
                                    if len(d) else None),
        "mean_valence_density": float(vd.mean()) if len(vd) else None,
        "valence_density_sd": float(vd.std()) if len(vd) else None,
        "valence_error_vs_target": (float(vd.mean() - cfg.valence_target)
                                    if len(vd) else None),
        **_coverage(el_counter, formulas, train_formulas, full_element_freq),
    }


def _coverage(el_counter, formulas, train_formulas, full_element_freq) -> dict:
    tot = sum(el_counter.values())
    if not tot:
        return {"distinct_elements": 0, "element_entropy_nats": None,
                "element_tv_vs_full_data": None, "unique_formula_fraction": None,
                "novel_formula_fraction": None}
    freq = {e: k / tot for e, k in el_counter.items()}
    ent = -sum(p * np.log(p) for p in freq.values() if p > 0)
    tv = None
    if full_element_freq:
        keys = set(freq) | set(full_element_freq)
        tv = 0.5 * sum(abs(freq.get(k, 0.0) - full_element_freq.get(k, 0.0)) for k in keys)
    uniq = len(set(formulas)) / len(formulas) if formulas else None
    novel = None
    if train_formulas is not None and formulas:
        novel = sum(f not in train_formulas for f in formulas) / len(formulas)
    return {
        "distinct_elements": len(freq),
        "element_entropy_nats": float(ent),
        "element_tv_vs_full_data": (float(tv) if tv is not None else None),
        "unique_formula_fraction": (float(uniq) if uniq is not None else None),
        "novel_formula_fraction": (float(novel) if novel is not None else None),
    }
