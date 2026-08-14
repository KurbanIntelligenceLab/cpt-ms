"""Priority 2: joint constraints at inference, and the attainability map.

WHAT IS NEW HERE AND WHAT IS NOT. The solver has always taken `A` of shape (B, m, V)
and returned m multipliers, and CP2 benchmarked its cost at m = 1, 2, 4, 8. So
multi-constraint capability is not new code. What has never been done is verifying that
the joint solve is CORRECT at m > 1 with the same protocol used at m = 1, and mapping
which pairs of targets are jointly attainable. Both are done here.

WHY AN ATTAINABILITY MAP IS THE INTERESTING ARTIFACT. Two constraints that are each
individually satisfiable need not be jointly satisfiable: the attainable set of a
distribution over a finite vocabulary is a convex polytope, and two linear functionals
can pin it to an empty intersection. A per-target error number cannot express that.
The map distinguishes three regimes, which is the result:

  jointly attainable      both targets inside the polytope; the dual solve converges
  individually only       each target reachable alone, no distribution meets both
  neither                 at least one target outside its own attainable range

WHAT THE VERIFICATION MUST RULE OUT. At m = 1 a wrong Jacobian still often produces a
plausible-looking projection, which is why CP2 used two independent oracles. The same
risk is worse at m > 1, because the multiplier Hessian is now a genuine m x m matrix
that can be near-singular when two constraint rows are nearly collinear. So this script
reports the Hessian condition number alongside the residual: a converged solve on a
badly conditioned system is not a trustworthy one, and reporting only the residual would
hide it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.layer.projection import project

SCRIPT = "scripts/layer/verify_multiconstraint.py"


def _attainable_range(a_row: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Min and max of a linear functional over the simplex on the allowed support.

    The extremes of a linear functional over a simplex are attained at vertices, so the
    range is simply the min and max coefficient among allowed tokens. Anything outside
    it is infeasible for a reason that has nothing to do with the solver.
    """
    vals = a_row[mask]
    return float(vals.min()), float(vals.max())


def joint_solve_case(B: int, V: int, m: int, seed: int, dtype=torch.float64) -> dict:
    """One randomized joint-constraint solve, verified the CP2 way.

    Targets are set from the statistic of the unconstrained distribution and then
    perturbed, so they sit inside the attainable range by construction: a convergence
    failure here is a solver problem, never an infeasible ask.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, V, generator=g, dtype=dtype)
    A = torch.randn(B, m, V, generator=g, dtype=dtype)
    p = torch.softmax(logits, dim=-1)
    b0 = torch.einsum("bmv,bv->bm", A, p)
    b = b0 + 0.15 * torch.randn(B, m, generator=g, dtype=dtype)

    res = project(logits, A, b, None, max_iter=100, tol=1e-12)
    q = res.q
    achieved = torch.einsum("bmv,bv->bm", A, q)
    residual = (achieved - b).abs()

    # Dual Hessian is the covariance of the constraint statistics under q. Its
    # conditioning bounds how well the multipliers are determined; a near-singular
    # Hessian means two constraints are nearly the same constraint.
    conds = []
    for i in range(B):
        Ai, qi = A[i], q[i]
        mu = Ai @ qi
        cov = (Ai * qi) @ Ai.T - torch.outer(mu, mu)
        conds.append(float(torch.linalg.cond(cov)))

    return {
        "B": B, "V": V, "m": m, "seed": seed,
        "all_converged": bool(res.converged.all()),
        "n_converged": int(res.converged.sum()),
        "iterations": int(res.iterations) if np.ndim(res.iterations) == 0 else None,
        "max_abs_residual": float(residual.max()),
        "mean_abs_residual": float(residual.mean()),
        "simplex_error": float((q.sum(-1) - 1).abs().max()),
        "min_mass": float(q.min()),
        "hessian_cond_max": float(max(conds)),
        "hessian_cond_median": float(np.median(conds)),
        "multiplier_absmax": float(res.lam.abs().max()),
    }


def attainability_grid(V: int, seed: int, n_grid: int = 9,
                       dtype=torch.float64) -> dict:
    """Map the joint feasible region of two constraints over one distribution.

    Constraint 1 is a mass-like positive statistic, constraint 2 a signed charge-like
    statistic, chosen to mirror the two families this project actually instruments
    (density and charge balance) rather than two arbitrary random rows: the question is
    whether a realistic PAIR composes, and two random Gaussian rows would be nearly
    orthogonal by construction and so unrealistically easy.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, V, generator=g, dtype=dtype)
    mass = torch.rand(V, generator=g, dtype=dtype) * 200.0 + 1.0     # strictly positive
    charge = torch.randint(-3, 4, (V,), generator=g).to(dtype)       # signed integers
    A = torch.stack([mass, charge]).unsqueeze(0)
    mask = torch.ones(1, V, dtype=torch.bool)

    r1 = _attainable_range(mass, mask[0])
    r2 = _attainable_range(charge, mask[0])
    # Grid strictly inside each marginal range: a point outside is trivially infeasible
    # and would tell us nothing about JOINT attainability, which is the question.
    g1 = np.linspace(r1[0] + 0.05 * (r1[1] - r1[0]), r1[1] - 0.05 * (r1[1] - r1[0]), n_grid)
    g2 = np.linspace(r2[0] + 0.05 * (r2[1] - r2[0]), r2[1] - 0.05 * (r2[1] - r2[0]), n_grid)

    cells = []
    for t1 in g1:
        for t2 in g2:
            b = torch.tensor([[t1, t2]], dtype=dtype)
            res = project(logits, A, b, mask, max_iter=200, tol=1e-12)
            ach = torch.einsum("bmv,bv->bm", A, res.q)
            resid = float((ach - b).abs().max())
            conv = bool(res.converged.all()) and resid < 1e-6
            # Individual attainability: solve each constraint alone.
            solo = []
            for j in range(2):
                rj = project(logits, A[:, j:j + 1, :], b[:, j:j + 1], mask,
                             max_iter=200, tol=1e-12)
                aj = torch.einsum("bmv,bv->bm", A[:, j:j + 1, :], rj.q)
                solo.append(bool(rj.converged.all()) and
                            float((aj - b[:, j:j + 1]).abs().max()) < 1e-6)
            cells.append({
                "mass_target": float(t1), "charge_target": float(t2),
                "joint_feasible": conv,
                "residual": resid,
                "each_alone_feasible": bool(solo[0] and solo[1]),
                "regime": ("jointly_attainable" if conv else
                           "individually_only" if (solo[0] and solo[1]) else
                           "neither"),
                "multiplier_absmax": float(res.lam.abs().max()),
            })

    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["regime"]] = counts.get(cell["regime"], 0) + 1
    return {
        "vocab_size": V,
        "mass_attainable_range": list(r1),
        "charge_attainable_range": list(r2),
        "n_grid": n_grid,
        "regime_counts": counts,
        "cells": cells,
        "note": "Grid points lie strictly inside each marginal attainable range, so any "
                "infeasibility recorded here is JOINT infeasibility rather than a target "
                "outside its own range.",
    }


def vocabulary_sweep(seed: int = 0, n_grid: int = 9, sizes=None) -> dict:
    """How joint attainability depends on vocabulary size.

    A single grid at one vocabulary size would have reported "everything composes",
    which is an artifact of a large vocabulary rather than a property of the method: the
    attainable polytope of a distribution over V tokens grows with V, so two constraints
    that pin a small vocabulary to an empty intersection have room to coexist in a large
    one. The dependence is the result; a single number would have concealed it.
    """
    out = []
    for V in (sizes or [8, 16, 32, 64, 128, 256, 512]):
        g = attainability_grid(V=V, seed=seed, n_grid=n_grid)
        n = len(g["cells"])
        rc = g["regime_counts"]
        out.append({
            "vocab_size": V,
            "n_cells": n,
            "jointly_attainable_fraction": rc.get("jointly_attainable", 0) / n,
            "individually_only_fraction": rc.get("individually_only", 0) / n,
            "neither_fraction": rc.get("neither", 0) / n,
        })
    return {
        "rows": out,
        "n_grid": n_grid,
        "caveat": "These fractions are GRID-DEPENDENT and must not be quoted as the "
                  "measure of a feasible region's size. Measured at V=64: 0.840 joint at "
                  "n_grid=5, 0.914 at 9, 0.938 at 15 -- a coarser grid places a larger "
                  "share of its points near the polytope's corners, where joint "
                  "infeasibility lives. The comparison ACROSS vocabulary sizes is valid "
                  "because every row uses the identical grid; the absolute level is not "
                  "a property of the constraint pair.",
    }


def main(smoke: bool = False) -> Path:
    V = 64 if smoke else 512
    seeds = [0] if smoke else [0, 1, 2]
    counts = [1, 2, 3, 4] if not smoke else [1, 2, 3]

    scaling = []
    for m in counts:
        for s in seeds:
            scaling.append(joint_solve_case(B=8, V=V, m=m, seed=s))

    grid = attainability_grid(V=V, seed=0, n_grid=(5 if smoke else 9))
    vsweep = vocabulary_sweep(seed=0, n_grid=(5 if smoke else 9),
                              sizes=([8, 64] if smoke else None))

    by_m: dict[int, list[dict]] = {}
    for row in scaling:
        by_m.setdefault(row["m"], []).append(row)
    summary = {
        "all_converged_at_every_m": all(r["all_converged"] for r in scaling),
        "worst_residual_by_m": {str(m): max(r["max_abs_residual"] for r in rs)
                                for m, rs in by_m.items()},
        "worst_hessian_cond_by_m": {str(m): max(r["hessian_cond_max"] for r in rs)
                                    for m, rs in by_m.items()},
        "constraint_counts_verified": counts,
        "joint_attainable_fraction_at_this_vocab": (
            grid["regime_counts"].get("jointly_attainable", 0) / len(grid["cells"])),
        "individually_only_fraction_at_this_vocab": (
            grid["regime_counts"].get("individually_only", 0) / len(grid["cells"])),
        "vocab_at_which_composition_becomes_free": next(
            (r["vocab_size"] for r in vsweep["rows"]
             if r["individually_only_fraction"] == 0.0), None),
        "smallest_vocab_joint_fraction": vsweep["rows"][0]["jointly_attainable_fraction"],
        "fractions_are_grid_dependent": True,
    }
    return emit(
        "CP6_multiconstraint" + ("_smoke" if smoke else ""),
        {
            "question": "Does the dual solve stay exact with several simultaneous "
                        "expectation constraints, and which target pairs are jointly "
                        "attainable?",
            "scaling": scaling,
            "attainability": grid,
            "vocabulary_sweep": vsweep,
            "summary": summary,
        },
        config={"smoke": smoke, "vocab": V, "constraint_counts": counts},
        seeds=seeds, script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(smoke=a.smoke))
