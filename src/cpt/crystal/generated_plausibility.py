from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from cpt.common.artifacts import emit, load
from cpt.crystal.metrics import is_structurally_valid
from cpt.crystal.mp20 import ATOMIC_MASS, CrystalVocab
from cpt.crystal.plausibility_screen import _smact_screens
from cpt.crystal.retarget import load_checkpoint
from cpt.train.train_crystal import CrystalTrainConfig, GrammarMask, sample

SCRIPT = "src/cpt/crystal/generated_plausibility.py"


def sequences_to_items(seqs, V):
    items = []
    n_parsed = 0
    for s in seqs:
        toks = [V.tokens[i] for i in s]
        ok = is_structurally_valid(toks)
        if not ok:
            continue
        n_parsed += 1
        formula, cur_el, lens, angs = {}, None, [], []
        for tok in toks:
            if tok.startswith("el:"):
                cur_el = tok[3:]
            elif tok.startswith("cnt:") and cur_el:
                formula[cur_el] = formula.get(cur_el, 0) + int(tok[4:])
                cur_el = None
            elif tok.startswith("len:"):
                j = int(tok[4:])
                lens.append(V.LEN_LO + (j + 0.5) * (V.LEN_HI - V.LEN_LO) / V.N_LEN)
            elif tok.startswith("ang:"):
                j = int(tok[4:])
                angs.append(V.ANG_LO + (j + 0.5) * (V.ANG_HI - V.ANG_LO) / V.N_ANG)
        vol = V.cell_volume(lens, angs) if len(lens) == 3 and len(angs) == 3 else None
        n_atoms = sum(formula.values())
        if not formula or vol is None or n_atoms == 0:
            continue
        if any(e not in ATOMIC_MASS for e in formula):
            continue
        items.append((formula, vol, n_atoms))
    return items, n_parsed


def screen_items(items, vpa_lo, vpa_hi):
    n = 0
    bal = ordered = packed = 0
    for formula, volume, n_atoms in items:
        n += 1
        b, o = _smact_screens(formula)
        bal += b
        ordered += o
        vpa = volume / n_atoms
        packed += (vpa_lo <= vpa <= vpa_hi)
    if not n:
        return {"n_screened": 0}
    return {
        "n_screened": n,
        "charge_balanced_rate": bal / n,
        "electronegativity_ordered_rate": ordered / n,
        "packing_plausible_rate": packed / n,
    }


def main(ckpt: str, n_samples: int = 2048, smoke: bool = False) -> Path:
    n_samples = 64 if smoke else n_samples
    ref = load("plausibility_reference")["payload"]
    lo = ref["packing_window_angstrom3_per_atom"]["lo_1pct"]
    hi = ref["packing_window_angstrom3_per_atom"]["hi_99pct"]
    reference_ceiling = ref["data_reference"]

    dev = torch.device("cpu")
    model, ck = load_checkpoint(ckpt, dev)
    cfg = CrystalTrainConfig(**ck["train_config"])
    V = CrystalVocab(ck["elements"], order=ck["vocab_order"])
    gm = GrammarMask(V, order=ck["vocab_order"])

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(0)
    seqs, diag = sample(model, V, gm, cfg, dev, n_samples, use_mask=True,
                        project_at_inference=True, max_len=ck["max_len"] + 2, rng=rng)

    n_sampled = len(seqs)
    items, n_parsed = sequences_to_items(seqs, V)
    n_built = len(items)
    screened = screen_items(items, lo, hi)

    funnel = {
        "n_sampled": n_sampled,
        "n_parsed_grammar_complete": n_parsed,
        "parsed_rate": n_parsed / n_sampled,
        "n_built_composition_cell": n_built,
        "built_given_parsed_rate": (n_built / n_parsed) if n_parsed else None,
        "built_rate_of_sampled": n_built / n_sampled,
        "screens_on_built": screened,
    }

    payload = {
        "checkpoint": Path(ckpt).name,
        "checkpoint_sha256": ck["checkpoint_sha256"],
        "tokenizer": ck["tokenizer"],
        "checkpoint_arm": ck["train_config"]["arm"],
        "checkpoint_density_target": ck["trained_density_target"],
        "sampling": {
            "n_samples": n_samples,
            "use_mask": True,
            "project_at_inference": True,
            "per_step_infeasible_fraction": diag["per_step_infeasible_fraction"],
            "degenerate_row_rate": diag["degenerate_row_rate"],
        },
        "funnel": funnel,
        "reference_ceiling": {
            "n_screened": reference_ceiling["n_screened"],
            "charge_balanced_rate": reference_ceiling["charge_balanced_rate"],
            "electronegativity_ordered_rate": reference_ceiling["electronegativity_ordered_rate"],
            "packing_plausible_rate": reference_ceiling["packing_plausible_rate"],
        },
        "packing_window_angstrom3_per_atom": {"lo_1pct": lo, "hi_99pct": hi},
        "min_interatomic_distance": {
            "computable": False,
            "reason": (
                "The generated serialization encodes composition (element, count) "
                "and cell (lengths, angles) tokens only; there are no fractional "
                "coordinate or Wyckoff-position tokens in the vocabulary "
                "(CrystalVocab defines only el:, cnt:, len:, ang: token families). "
                "Minimum interatomic distance requires atomic positions, which the "
                "model does not emit and does not attempt to predict."
            ),
        },
        "dft_subset": {
            "attempted": False,
            "reason": ref["why_not_energy"],
        },
        "note": (
            "charge_balanced_rate, electronegativity_ordered_rate and "
            "packing_plausible_rate are computed by the IDENTICAL screen path "
            "(src/cpt/crystal/plausibility_screen.py::_smact_screens and the same "
            "packing window) used for the reference, applied to generated "
            "compositions and cells that parsed and built."
        ),
        "scope_note": (
            'The generator emits composition and cell parameters without atomic positions. Grammar completion, composition-cell decoding and chemical screens use distinct denominators; these screens do not establish structural stability.'
        ),
    }
    return emit(
        "generated_plausibility" + ("_smoke" if smoke else ""),
        payload,
        config={"n_samples": n_samples, "smoke": smoke, "checkpoint": Path(ckpt).name},
        seeds=[0], script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", required=True, help="trained density checkpoint")
    a = ap.parse_args()
    print(main(ckpt=a.ckpt, n_samples=a.n_samples, smoke=a.smoke))
