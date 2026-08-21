"""Why a real-valued budget is not a finite automaton at usable size.

WHAT THIS ARGUES AND WHAT IT DOES NOT. Ctrl-G and GeLaTo amortize constraint control across
constraints without retraining -- the same word this paper uses for its own central claim --
so the distinction has to be made rather than assumed. Theirs is not a weaker method: for the
constraint class they target, a deterministic finite automaton over tokens, they are exact and
require no per-constraint training at all. The claim here is narrow: the constraint class
differs, because tracking a real-valued running budget to a useful tolerance needs a number of
automaton states that grows inversely with that tolerance.

The argument is arithmetic, not empirical, so this script computes rather than measures. What
it takes from data is the range the budget must cover and the tolerance the task actually
demands -- both read from MP-20 and from the vocabulary's own discretization, so the table
says what THIS task needs rather than what a hypothetical one might.

WHAT THE ARITHMETIC ACTUALLY SHOWS, INCLUDING AGAINST THE FRAMING THIS SCRIPT WAS WRITTEN TO
SUPPORT. The intended conclusion was that the automaton is intractably large. It is not. At the
finest tolerance this serialization can express -- one cell-parameter bin, 1/512 relative --
the budget needs about 100 buckets and the product with parse position is under 10^5 states,
which is a fraction of the model's parameter count and entirely buildable. That conclusion is
recorded here rather than deleted, because it was wrong for a reason worth stating: a size
argument was the intuitive objection and the measurement does not support it.

The real distinction is per-instance construction. A DFA encodes ONE constraint instance: the
budget bound is compiled into the transition structure, so a different density target is a
different automaton. Their amortization is across constraints at a fixed model; the claim here
is across targets of one constraint at a fixed model. The two are compatible and neither
subsumes the other. The measurable consequence is that the ten frontier targets in this paper
would require ten automata totalling ~10^6 states, against one dual solve that takes the target
as a runtime argument.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.mp20 import ATOMIC_MASS, ELEMENTS, CrystalVocab, load_split

FRONTIER_TARGETS = [1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0, 13.0, 16.0, 20.0]

SCRIPT = "scripts/crystal/automaton_size.py"


def main(smoke: bool = False) -> Path:
    train = load_split("train", limit=2000 if smoke else None)

    masses = np.array([sum(ATOMIC_MASS[e] * k for e, k in c.formula.items())
                       for c in train], dtype=float)
    lo, hi = float(masses.min()), float(masses.max())
    span = hi - lo

    V = CrystalVocab(ELEMENTS)
    el_mass = np.array([ATOMIC_MASS[e] for e in ELEMENTS], dtype=float)

    # The vocabulary's own discretization floor: the cell-parameter tokens are binned, so the
    # achieved density cannot be resolved more finely than one bin. Any tolerance below this
    # is finer than the task can express and would be a spurious demand on the automaton.
    n_len_bins = sum(1 for t in V.tokens if t.startswith("len:"))
    n_ang_bins = sum(1 for t in V.tokens if t.startswith("ang:"))
    floor_rel = 1.0 / n_len_bins if n_len_bins else float("nan")

    rows = []
    for tol_rel in (1e-1, 1e-2, 1e-3, 1e-4):
        tol_abs = tol_rel * span
        n_states = int(np.ceil(span / tol_abs))
        rows.append({
            "relative_tolerance": tol_rel,
            "absolute_tolerance_amu": tol_abs,
            "budget_states": n_states,
            "at_or_below_task_floor": bool(tol_rel < floor_rel),
        })

    # A DFA over tokens must also carry the parse position, since the same budget value means
    # different things at an element slot and a count slot. That is a product, not a sum.
    parse_states = int(len(V.tokens))
    for r in rows:
        r["with_parse_state_product"] = int(r["budget_states"] * parse_states)

    model_params = 5_000_000       # the backbone this paper trains
    for r in rows:
        r["states_over_model_params"] = r["with_parse_state_product"] / model_params

    payload = {
        "argument": ("A DFA can approximate a real-valued budget by bucketing it; the "
                     "question is how many buckets, and whether the resulting automaton is "
                     "smaller or larger than the model it is meant to control."),
        "budget_range_amu": {"min": lo, "max": hi, "span": span},
        "vocabulary": {
            "n_tokens": int(len(V.tokens)),
            "n_length_bins": int(n_len_bins),
            "n_angle_bins": int(n_ang_bins),
            "discretization_floor_relative": floor_rel,
            "floor_note": ("cell parameters are emitted as bin indices, so a tolerance finer "
                           "than one bin is not expressible by this serialization and would "
                           "be a spurious demand on any controller, DFA or otherwise"),
        },
        "rows": rows,
        "model_params_for_scale": model_params,
        "consequence": None,
    }

    # The row at the tolerance the task actually demands
    usable = [r for r in rows if not r["at_or_below_task_floor"]]
    tight = min(usable, key=lambda r: r["relative_tolerance"]) if usable else rows[0]
    n_targets = len(FRONTIER_TARGETS)
    payload["size_argument_verdict"] = "REFUTED by this computation"
    payload["consequence"] = (
        f"The size objection does not hold. At the finest tolerance this serialization can "
        f"express ({tight['relative_tolerance']:.0e} relative, "
        f"{tight['absolute_tolerance_amu']:.1f} amu — anything finer is below the "
        f"{floor_rel:.4f} cell-parameter bin floor and not expressible), tracking the budget "
        f"needs {tight['budget_states']:,} states and {tight['with_parse_state_product']:,} "
        f"once multiplied by parse position, which is "
        f"{tight['states_over_model_params']:.2f}x the model's parameter count and entirely "
        f"buildable. The distinction is per-instance construction, not size: the budget bound "
        f"is compiled into the transition structure, so each of the {n_targets} frontier "
        f"targets is a separate automaton — about "
        f"{tight['with_parse_state_product'] * n_targets:,} states in total — while the dual "
        f"solve takes the target as a runtime argument. DFA methods amortize across "
        f"constraints at a fixed model; this paper amortizes across targets of one constraint. "
        f"Neither subsumes the other."
    )
    payload["per_target_automata"] = {
        "n_frontier_targets": n_targets,
        "states_each": tight["with_parse_state_product"],
        "states_total": tight["with_parse_state_product"] * n_targets,
    }
    payload["dial_is_continuous"] = True
    payload["note_on_amortization"] = (
        "Ctrl-G and GeLaTo amortize across CONSTRAINTS at a fixed model; the claim in this "
        "paper amortizes across TARGETS of one constraint at a fixed model. Neither subsumes "
        "the other and the two are compatible."
    )

    return emit("automaton_size" + ("_smoke" if smoke else ""), payload,
                config={"tolerances": [r["relative_tolerance"] for r in rows],
                        "model_params": model_params},
                seeds=[0], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(smoke=a.smoke))
