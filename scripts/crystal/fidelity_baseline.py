"""Rejection-accepted versus steered samples at the same target."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.mp20 import ATOMIC_MASS, CrystalVocab, ELEMENTS, load_split

SCRIPT = "scripts/crystal/fidelity_baseline.py"


def _composition_profile(crystals) -> dict:
    el = Counter()
    n_el, n_at, mass = [], [], []
    for c in crystals:
        for e, k in c.formula.items():
            el[e] += k
        n_el.append(len(c.formula))
        n_at.append(c.n_atoms)
        mass.append(c.mass)
    tot = sum(el.values()) or 1
    return {
        "n_samples": len(crystals),
        "distinct_elements": len(el),
        "top_elements": [(e, round(k / tot, 4)) for e, k in el.most_common(10)],
        "element_frequencies": {e: k / tot for e, k in el.items()},
        "mean_distinct_elements_per_structure": float(np.mean(n_el)) if n_el else None,
        "mean_atoms": float(np.mean(n_at)) if n_at else None,
        "mean_mass_u": float(np.mean(mass)) if mass else None,
    }


def _tv(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def rejection_arm(crystals, target: float, tol: float) -> dict:
    """What an idealized rejection sampler returns at this target."""
    acc = [c for c in crystals if abs(c.density - target) <= tol]
    n = len(crystals)
    out = {
        "target": target,
        "tolerance": tol,
        "n_accepted": len(acc),
        "acceptance_rate": len(acc) / n if n else 0.0,
        "expected_draws": (n / len(acc)) if acc else None,
        "reachable": bool(acc),
    }
    if acc:
        dens = np.array([c.density for c in acc])
        out |= {
            "mean_density": float(dens.mean()),
            "density_sd": float(dens.std()),
            "composition": _composition_profile(acc),
        }
    return out


def main(targets: list[float], tol: float = 0.5, smoke: bool = False) -> Path:
    train = load_split("train", limit=3000 if smoke else None)
    V = CrystalVocab(ELEMENTS, order="cell_first")
    all_prof = _composition_profile(train)

    rows = {}
    for t in targets:
        arm = rejection_arm(train, t, tol)
        if arm["reachable"]:
            arm["tv_vs_full_data"] = _tv(
                arm["composition"]["element_frequencies"],
                all_prof["element_frequencies"],
            )
        rows[f"{t:g}"] = arm

    reachable = [k for k, v in rows.items() if v["reachable"]]
    unreachable = [k for k, v in rows.items() if not v["reachable"]]
    draws = [v["expected_draws"] for v in rows.values() if v["expected_draws"]]

    payload = {
        "tolerance": tol,
        "n_train": len(train),
        "targets": targets,
        "full_data_composition": all_prof,
        "rejection_arm": rows,
        "summary": {
            "targets_reachable_by_rejection": len(reachable),
            "targets_unreachable_by_rejection": unreachable,
            "min_expected_draws": float(min(draws)) if draws else None,
            "max_expected_draws": float(max(draws)) if draws else None,
            "note": (
                "The rejection arm draws from the EMPIRICAL data, i.e. it assumes a "
                "generator reproducing the data distribution perfectly. mask-only does not "
                "achieve that, so any difference measured here understates the "
                "difference against a real baseline. Steered-sample profiles are "
                "filled in from the frontier runs; this artifact establishes what "
                "the baseline returns before those land."
            ),
        },
    }
    return emit(
        "fidelity_baseline" + ("_smoke" if smoke else ""),
        payload,
        config={"targets": targets, "tolerance": tol, "smoke": smoke},
        seeds=[], script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args()
    tg = [1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0, 13.0, 16.0, 20.0]
    print(main(targets=tg, tol=a.tol, smoke=a.smoke))
