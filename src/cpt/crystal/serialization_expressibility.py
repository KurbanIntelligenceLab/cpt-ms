"""Why the second serialization is not a training run: the constraint does not exist there."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cpt.common.artifacts import emit
from cpt.crystal.mp20 import ATOMIC_MASS, ELEMENTS, VALENCE, CrystalVocab, load_split

SCRIPT = "src/cpt/crystal/serialization_expressibility.py"
ORDERS = ("cell_first", "comp_first")


def main(n_structures: int = 400, smoke: bool = False) -> Path:
    from cpt.crystal.channels import _additive_budget
    from cpt.crystal.model import CrystalTrainConfig

    train = load_split("train", limit=200 if smoke else n_structures)
    cfg = CrystalTrainConfig(arm="projected", density_target=5.0)

    rows = []
    for order in ORDERS:
        V = CrystalVocab(ELEMENTS, order=order)
        enc = [V.encode(c) for c in train]
        det = tot = 0
        det_val = tot_val = 0
        for e in enc:
            ids = np.array([e], dtype=np.int64)
            for cut in range(2, len(e)):
                tok = V.tokens[e[cut]]
                if not tok.startswith("el:"):
                    continue
                pre = ids[:, :cut]
                b, slots = _additive_budget(pre, V, cfg, ATOMIC_MASS,
                                            cfg.density_target, 1.6605390666)
                tot += 1
                if np.isfinite(b[0]) and slots[0] > 0:
                    det += 1
                bv, sv = _additive_budget(pre, V, cfg, {k: float(v) for k, v in VALENCE.items()},
                                          0.48, 1.0)
                tot_val += 1
                if np.isfinite(bv[0]) and sv[0] > 0:
                    det_val += 1
        rows.append({
            "order": order,
            "n_element_positions": tot,
            "n_budget_determined": det,
            "determined_fraction": det / tot if tot else None,
            "n_valence_positions": tot_val,
            "n_valence_determined": det_val,
            "valence_determined_fraction": det_val / tot_val if tot_val else None,
        })

    good = [r for r in rows if (r["determined_fraction"] or 0) > 0.5]
    bad = [r for r in rows if (r["determined_fraction"] or 0) <= 1e-9]

    payload = {
        "question": ("Can the density constraint be expressed as a per-step linear "
                     "expectation under each serialization?"),
        "n_structures": len(train),
        "density_target_used": cfg.density_target,
        "rows": rows,
        "mechanism": (
            "Density is a ratio, so a target density fixes a total mass budget only once "
            "the cell volume is known. Cell-first closes the volume before any element "
            "token, making the running budget a determined real number; composition-first "
            "leaves it undetermined while the element tokens are emitted, so there is no "
            "budget to project onto."),
        "consequence_for_the_requested_experiment": None,
        "verdict": None,
    }

    if good and bad:
        payload["verdict"] = (
            f"The constraint is expressible under {good[0]['order']} at "
            f"{good[0]['determined_fraction'] * 100:.1f}% of element positions and under "
            f"{bad[0]['order']} at {bad[0]['determined_fraction'] * 100:.1f}%. The same "
            f"holds for the second family "
            f"({good[0]['valence_determined_fraction'] * 100:.1f}% against "
            f"{bad[0]['valence_determined_fraction'] * 100:.1f}%), since it is also a "
            f"per-volume density.")
        payload["consequence_for_the_requested_experiment"] = (
            f"Retraining the density arm under {bad[0]['order']} would train with zero "
            f"actionable positions, so it would measure the unconstrained model rather than "
            f"a serialization effect. The limitation is therefore not that this experiment "
            f"was skipped: the constraint class and the serialization are COUPLED, and the "
            f"coupling is the finding. A serialization comparison for this method requires "
            f"two orderings that both determine the budget before the constrained tokens -- "
            f"a different axis of variation than the one the determinability study used.")
    else:
        payload["verdict"] = (
            "Both orderings determine the budget at similar rates, so a serialization "
            "comparison is a straightforward training run after all.")

    return emit("serialization_expressibility" + ("_smoke" if smoke else ""), payload,
                config={"orders": list(ORDERS), "n_structures": len(train),
                        "density_target": cfg.density_target},
                seeds=[0], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-structures", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(n_structures=a.n_structures, smoke=a.smoke))
