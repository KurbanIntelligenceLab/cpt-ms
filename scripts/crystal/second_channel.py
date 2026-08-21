"""A second constraint family on real crystals, and why it is not charge neutrality.

WHY NOT CHARGE NEUTRALITY. The plan's family (i) is charge balance, and the obvious move
is to make it a Regime E target so the paper has a constraint the model does not directly
emit. It cannot be done with this layer, and the reason is structural rather than an
implementation gap.

The layer constrains a LINEAR functional of the next token: sum_i a_i q_i = b. Charge
neutrality as SMACT defines it -- and as this project's own plausibility screen measures
it, at a real-data rate of 0.7150 -- is not a linear functional. It is an EXISTENTIAL
predicate over oxidation-state assignments: a composition is balanced if THERE EXISTS an
assignment of formal charges, drawn from each element's allowed states, summing to zero.
In MP-20's training split 71 of 85 elements carry more than one allowed oxidation state,
and 81.8% of element-occurrences involve a multivalent element, so the existential
quantifier is doing nearly all the work.

Collapsing it to one charge per element makes it linear and destroys it. Measured on 4,000
training structures: SMACT calls 71.07% balanced, the linear proxy using each element's
first-listed oxidation state calls 7.95% balanced, the two agree on 36.78% of structures,
and the proxy recovers only 11.12% of the SMACT-balanced set -- 2,527 false negatives
against 2 false positives. A Regime E target built on that proxy would be steering a
quantity that has almost nothing to do with the screen the paper reports.

So charge neutrality is recorded as OUTSIDE the expressible class, which is a scope result
worth stating plainly, and the second family is chosen to be genuinely linear.

WHAT THE SECOND FAMILY IS. Valence electron count per unit volume: sum over the formula of
each element's valence electron count, divided by the cell volume. It is additive in
element counts, so it fits the channel exactly; it is defined for every element in the
split (100% computable, against 98.9% for mean electronegativity); it is chemically
meaningful, being the quantity electron-count rules are written in; and it is NOT directly
emitted -- the model emits elements, counts and lattice parameters, never an electron
count.

Critically it is not a restatement of density. Across the 26,910 training structures the
correlation between mass density and valence electron density is 0.725. That matters
because the multiplier Hessian is the covariance of the constraint statistics, so two
nearly collinear constraints make the joint solve ill-posed -- the pathology this
project's own multi-constraint tests construct deliberately. A correlation of 0.725 is
high enough that the pair is a real test of composition and low enough to stay
conditioned, which is measured here rather than assumed.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.mp20 import load_split
from scripts.layer.projection import project
from scripts.train.train_crystal import (
    ATOMIC_MASS,
    ELEMENTS,
    CrystalVocab,
    weight_channel,
)

SCRIPT = "scripts/crystal/second_channel.py"

# amu/A^3 -> g/cm^3, the same conversion the density channel uses.
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


def valence_table(elements=None) -> dict[str, int]:
    """Valence electron count per element, from SMACT."""
    from smact import Element
    out = {}
    for e in (elements or ELEMENTS):
        try:
            v = Element(e).num_valence
        except Exception:
            continue
        if v is not None:
            out[e] = int(v)
    return out


def charge_linearity_evidence(n_structures: int = 4000) -> dict:
    """Measure how badly a linear proxy approximates the SMACT charge screen.

    This is the evidence for the scope claim, not an assertion of it. If a future
    serialization or oxidation-state table made the proxy adequate, this measurement
    would say so.
    """
    from smact import Element
    train = load_split("train")
    els = sorted({e for c in train for e in c.formula})
    states = {}
    for e in els:
        try:
            s = Element(e).oxidation_states
        except Exception:
            s = ()
        states[e] = tuple(s) if s else ()

    multivalent = sum(1 for e in els if len(states[e]) > 1)
    with_states = sum(1 for e in els if states[e])
    occ_multi = sum(1 for c in train for e in c.formula if len(states.get(e, ())) > 1)
    occ_total = sum(1 for c in train for _ in c.formula)

    def smact_balanced(formula) -> bool:
        ox = [states[e] for e in formula]
        if any(not o for o in ox):
            return False
        counts = [formula[e] for e in formula]
        for st in itertools.product(*ox):
            if sum(s * c for s, c in zip(st, counts)) == 0:
                return True
        return False

    first = {e: (states[e][0] if states[e] else 0) for e in els}
    sub = train[:n_structures]
    truth = np.array([smact_balanced(c.formula) for c in sub])
    proxy = np.array([sum(first.get(e, 0) * n for e, n in c.formula.items()) == 0
                      for c in sub])
    tp = int((truth & proxy).sum())
    return {
        "n_structures": len(sub),
        "n_elements": len(els),
        "n_elements_with_states": with_states,
        "n_multivalent_elements": multivalent,
        "multivalent_occurrence_fraction": occ_multi / occ_total,
        "smact_balanced_rate": float(truth.mean()),
        "linear_proxy_balanced_rate": float(proxy.mean()),
        "agreement": float((truth == proxy).mean()),
        "recall_of_smact_balanced": tp / max(int(truth.sum()), 1),
        "false_positives": int(((~truth) & proxy).sum()),
        "false_negatives": int((truth & (~proxy)).sum()),
        "verdict": "charge neutrality is an existential predicate over oxidation-state "
                   "assignments, not a linear functional; it is outside the class this "
                   "layer can express",
    }


def channel_statistics() -> dict:
    """Data statistics for the two channels, including their collinearity."""
    train = load_split("train")
    VE = valence_table(sorted({e for c in train for e in c.formula}))
    mass_d, val_d = [], []
    for c in train:
        vol = c.volume
        if vol is None:
            continue
        m = sum(ATOMIC_MASS.get(e, 0.0) * n for e, n in c.formula.items())
        v = sum(VE.get(e, 0) * n for e, n in c.formula.items())
        mass_d.append(m / vol * AMU_PER_A3_TO_G_PER_CM3)
        val_d.append(v / vol)
    a, b = np.array(mass_d), np.array(val_d)
    return {
        "n_structures": len(a),
        "n_elements_with_valence": len(VE),
        "mass_density": {"mean": float(a.mean()), "sd": float(a.std()),
                         "min": float(a.min()), "max": float(a.max())},
        "valence_electron_density": {"mean": float(b.mean()), "sd": float(b.std()),
                                     "min": float(b.min()), "max": float(b.max())},
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "units": {"mass_density": "g/cm^3", "valence_electron_density": "e/A^3"},
    }


def _rank_ratio(cm: np.ndarray, cv: np.ndarray) -> float:
    """Second singular value over the first for the 2-row channel matrix.

    Zero means the two channels are the same constraint on this support, so the joint
    solve is asking for one constraint twice and its Hessian is singular by construction
    rather than by numerical accident.
    """
    s = np.linalg.svd(np.stack([cm, cv]), compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


def joint_conditioning(n_prefixes: int = 64, seed: int = 0) -> dict:
    """Solve BOTH constraints jointly on real crystal prefixes and report conditioning.

    Targets are set from the statistic of the unconstrained masked distribution and
    perturbed, so they sit inside the attainable range by construction -- any failure is
    the solver's, not an infeasible ask. This is the crystal-data counterpart of the
    synthetic multi-constraint check.
    """
    from scripts.train.train_crystal import CrystalTrainConfig, GrammarMask

    train = load_split("train", limit=2000)
    V = CrystalVocab(ELEMENTS, order="cell_first")
    gm = GrammarMask(V, order="cell_first")
    VE = valence_table()
    enc = [V.encode(c) for c in train[:n_prefixes]]
    # Truncate to a length where the grammar still offers a CHOICE. Past the end of a
    # sequence only <EOS> is legal, so a prefix cut there has a one-token mask and the
    # constraint has nothing to act on: the attainable range is a single point and the
    # solve is vacuous. Measured allowed-token counts by position on this serialization
    # are 512 at the lattice slots, 94 and 16 alternating at element and count slots, and
    # 1 past the end -- so prefixes are cut at each row's own element/count positions
    # rather than at a fixed index.
    def next_token_mask(prefix: np.ndarray) -> np.ndarray:
        """Mask for the token about to be APPENDED after `prefix`.

        `masks_for` returns the mask AT each existing position, so index -1 is the mask
        for the token already emitted. Taking it directly describes the wrong position:
        the channel would be evaluated for the next token while the mask constrained the
        previous one, and the two would disagree about which slot is being filled. The
        sampling loop documents the same trap and solves it the same way -- pad by one
        column and take the last. Getting this wrong is not a small error: it silently
        produced a support on which the mass channel was identically zero, i.e. an
        attainable range of {0}, which then read as ill-conditioning.
        """
        padded = np.concatenate(
            [prefix, np.zeros((prefix.shape[0], 1), dtype=prefix.dtype)], axis=1)
        return gm.masks_for(padded)[0, -1, :]

    A_rows, mask_rows, kinds = [], [], []
    n_collinear_positions = n_independent_positions = 0
    rng = np.random.default_rng(seed)
    val_tbl = {k: float(v) for k, v in VE.items()}
    for i, e in enumerate(enc):
        row = np.array(e, dtype=np.int64)[None, :]
        cand = []
        for cut in range(3, len(e)):
            pre = row[:, :cut]
            mk = next_token_mask(pre)
            if mk.sum() <= 1:
                continue
            # Require the constraint to be ACTIONABLE at this position: at least two
            # allowed tokens must carry different channel values, or no substitution can
            # move the constrained quantity and the solve is vacuous regardless of
            # conditioning. This is the actionability boundary WS-2 measured, applied as
            # a precondition on which prefixes can test the joint solve at all.
            cm = weight_channel(pre, V, ATOMIC_MASS)[0][mk]
            cv = weight_channel(pre, V, val_tbl)[0][mk]
            if np.unique(cm).size > 1 and np.unique(cv).size > 1:
                cand.append((cut, _rank_ratio(cm, cv)))
        if not cand:
            continue
        # Split by whether the two channels are linearly independent on this support.
        # At a COUNT position with pending element E the two channels are MASS[E]*k and
        # VALENCE[E]*k over the tokens cnt:k, so both are proportional to k and the pair
        # is exactly collinear: one constraint, written twice. Only ELEMENT positions
        # carry rank 2. This is a property of the serialization, and it is the answer to
        # "where can two additive-per-element quantities be steered jointly".
        indep = [cut for cut, rr in cand if rr > 1e-6]
        collin = [cut for cut, rr in cand if rr <= 1e-6]
        n_collinear_positions += len(collin)
        n_independent_positions += len(indep)
        if not indep:
            continue
        cut = int(rng.choice(indep))
        pre = row[:, :cut]
        A_rows.append(np.stack([weight_channel(pre, V, ATOMIC_MASS)[0],
                                weight_channel(pre, V, val_tbl)[0]]))
        mask_rows.append(next_token_mask(pre))
        kinds.append("count" if all(V.tokens[i].startswith("cnt:")
                                    for i in np.where(mask_rows[-1])[0]) else "element")
    if not A_rows:
        raise RuntimeError("no prefix offered a grammatical choice; the mask or the "
                           "serialization changed")
    A = torch.tensor(np.stack(A_rows), dtype=torch.float64)
    mask_t = torch.tensor(np.stack(mask_rows))
    L = int(A.shape[0])
    B = int(A.shape[0])
    logits = torch.tensor(rng.normal(size=(B, len(V))), dtype=torch.float64)

    logp = logits.masked_fill(~mask_t, float("-inf")).log_softmax(-1)
    p = logp.exp()
    b0 = torch.einsum("bmv,bv->bm", A, p)
    b = b0 * torch.tensor(1.0 + 0.10 * rng.normal(size=(B, 2)), dtype=torch.float64)

    res = project(logits, A, b, mask_t, max_iter=200, tol=1e-12)
    q = res.q
    ach = torch.einsum("bmv,bv->bm", A, q)
    resid = (ach - b).abs()

    conds, singular = [], 0
    for i in range(B):
        mu = A[i] @ q[i]
        cov = (A[i] * q[i]) @ A[i].T - torch.outer(mu, mu)
        cnd = float(torch.linalg.cond(cov))
        if not np.isfinite(cnd) or cnd > 1e12:
            singular += 1
        else:
            conds.append(cnd)
    return {
        "n_prefixes_with_choice": int(B),
        "n_prefixes_offered": len(enc),
        "position_kinds_used": {k: kinds.count(k) for k in set(kinds)},
        "n_rank2_positions_found": n_independent_positions,
        "n_collinear_positions_found": n_collinear_positions,
        "collinear_position_explanation": (
            "At a count position with pending element E both channels are proportional "
            "to the count k (MASS[E]*k and VALENCE[E]*k), so the pair is exactly "
            "collinear and the joint solve is one constraint written twice. Rank 2 "
            "requires an element position, where the two tables differ across elements."
        ),
        "all_converged": bool(res.converged.all()),
        "n_converged": int(res.converged.sum()),
        "max_abs_residual": float(resid.max()),
        "mean_abs_residual": float(resid.mean()),
        "simplex_error": float((q.sum(-1) - 1).abs().max()),
        "mass_outside_grammar_mask": float(q.masked_fill(mask_t, 0.0).sum(-1).max()),
        "hessian_cond_median": float(np.median(conds)) if conds else None,
        "hessian_cond_p95": float(np.percentile(conds, 95)) if conds else None,
        "n_ill_conditioned": singular,
    }


def main(smoke: bool = False) -> Path:
    payload = {
        "question": "Can a second constraint family be steered on real crystals jointly "
                    "with density, and is charge neutrality expressible at all?",
        "charge_is_not_linear": charge_linearity_evidence(400 if smoke else 4000),
        "channel_statistics": channel_statistics(),
        "joint_conditioning": joint_conditioning(16 if smoke else 64),
        "second_family": {
            "name": "valence_electron_density",
            "definition": "sum of per-element valence electron counts over the formula, "
                          "divided by cell volume",
            "units": "electrons per cubic angstrom",
            "why_not_charge": "charge neutrality is existential over oxidation states, "
                              "not linear; see charge_is_not_linear",
            "directly_emitted": False,
        },
    }
    return emit("second_channel" + ("_smoke" if smoke else ""), payload,
                config={"smoke": smoke, "second_family": "valence_electron_density"},
                seeds=[0], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(smoke=a.smoke))
