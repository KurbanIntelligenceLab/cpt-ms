"""Physical plausibility of generated crystals, without atomic positions."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from scripts.common.artifacts import emit
from scripts.crystal.mp20 import load_split

SCRIPT = "scripts/crystal/plausibility_screen.py"


def _smact_screens(formula: dict[str, int]) -> tuple[bool, bool]:
    from smact import Element, neutral_ratios
    from smact.screening import pauling_test
    try:
        els = [Element(e) for e in formula]
    except Exception:
        return False, False
    counts = tuple(int(formula[e]) for e in formula)
    ox_combos = [e.oxidation_states for e in els]
    if any(not o for o in ox_combos):
        return False, False
    import itertools
    balanced = False
    ordered = False
    for states in itertools.product(*ox_combos):
        if sum(s * c for s, c in zip(states, counts)) != 0:
            continue
        balanced = True
        try:
            electronegs = [e.pauling_eneg for e in els]
            if None in electronegs:
                continue
            if pauling_test(states, electronegs):
                ordered = True
                break
        except Exception:
            continue
    return balanced, ordered


def screen_compositions(items, vpa_lo: float, vpa_hi: float, limit: int | None = None):
    """Run all three screens over (formula, volume, n_atoms) triples."""
    n = 0
    bal = ordered = packed = 0
    for formula, volume, n_atoms in items[:limit] if limit else items:
        if not formula or not n_atoms or volume is None or volume <= 0:
            continue
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


def main(limit: int = 500, smoke: bool = False) -> Path:
    limit = 40 if smoke else limit
    train = load_split("train")
    vpa = np.array([c.volume / c.n_atoms for c in train])
    lo, hi = float(np.percentile(vpa, 1)), float(np.percentile(vpa, 99))

    rng = np.random.default_rng(0)
    pick = rng.choice(len(train), size=min(limit, len(train)), replace=False)
    data_items = [(train[i].formula, train[i].volume, train[i].n_atoms) for i in pick]
    data_ref = screen_compositions(data_items, lo, hi)

    payload = {
        "screens": ["charge_neutrality", "electronegativity_order", "packing_plausibility"],
        "why_not_energy": (
            "Generated serialization is composition + cell with no atomic positions. An "
            "energy model requires positions; obtaining them is crystal structure "
            "prediction, which the model does not attempt. Relaxing from random "
            "positions would measure the initialization, not the generated crystal. The "
            "calibrated CHGNet screen therefore applies to reference structures only."
        ),
        "packing_window_a3_per_atom": {
            "lo_1pct": lo, "hi_99pct": hi,
            "data_mean": float(vpa.mean()), "data_sd": float(vpa.std()),
        },
        "data_reference": data_ref,
        "note": (
            "The data reference is scored through the IDENTICAL path as any generated "
            "sample, so it is a measured ceiling rather than an assumed 100%."
        ),
    }
    return emit(
        "plausibility_reference" + ("_smoke" if smoke else ""),
        payload,
        config={"limit": limit, "smoke": smoke}, seeds=[0], script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(limit=a.limit, smoke=a.smoke))
