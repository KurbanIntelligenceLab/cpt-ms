from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cpt.common.artifacts import emit
from cpt.crystal.mp20 import ATOMIC_MASS, ELEMENTS, VALENCE, CrystalVocab
from cpt.layer.projection import ProjectionLayer, project

SCRIPT = "src/cpt/layer/robustness_sweep.py"

VOCAB = CrystalVocab(ELEMENTS)
VSIZE = len(VOCAB)

MASS_ROW = torch.zeros(VSIZE, dtype=torch.float64)
VALENCE_ROW = torch.zeros(VSIZE, dtype=torch.float64)
EL_MASK = torch.zeros(VSIZE, dtype=torch.bool)
for _e in ELEMENTS:
    _i = VOCAB.index[f"el:{_e}"]
    MASS_ROW[_i] = ATOMIC_MASS[_e]
    VALENCE_ROW[_i] = float(VALENCE[_e])
    EL_MASK[_i] = True


def make_case(B: int, m: int, seed: int, collinear: bool = False):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, VSIZE, generator=g, dtype=torch.float64)
    mask = EL_MASK.unsqueeze(0).expand(B, VSIZE).clone()
    rows = [MASS_ROW]
    if m == 2:
        if collinear:
            noise = torch.randn(VSIZE, generator=g, dtype=torch.float64) * 1e-4
            rows.append(MASS_ROW * 1.0003 + noise * EL_MASK.to(torch.float64))
        else:
            rows.append(VALENCE_ROW)
    elif m > 2:
        raise ValueError(f"unsupported m={m}")
    A = torch.stack(rows).unsqueeze(0).expand(B, m, VSIZE).clone()
    p = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    b0 = torch.einsum("bmv,bv->bm", A, p)
    b = b0 + 0.2 * torch.randn(B, m, generator=g, dtype=torch.float64) * b0.abs().clamp_min(1.0)
    return logits, A, b, mask


def attainable_range(row: torch.Tensor, mask_row: torch.Tensor) -> tuple[float, float]:
    vals = row[mask_row]
    return float(vals.min()), float(vals.max())


def tolerance_sweep(m: int, B: int, tols, max_iter: int, seed: int, ref_tol: float) -> dict:
    logits, A, b, mask = make_case(B, m, seed)
    w = torch.randn(B, VSIZE, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)

    ref = project(logits, A, b, mask, max_iter=max(max_iter, 500), tol=ref_tol)
    x_ref = logits.clone().requires_grad_(True)
    (ProjectionLayer.apply(x_ref, A, b, mask, max(max_iter, 500), ref_tol) * w).sum().backward()
    grad_ref = x_ref.grad.detach()

    rows = []
    for tol in tols:
        t0 = time.perf_counter()
        res = project(logits, A, b, mask, max_iter=max_iter, tol=tol)
        wall_forward = time.perf_counter() - t0

        x = logits.clone().requires_grad_(True)
        t0 = time.perf_counter()
        q = ProjectionLayer.apply(x, A, b, mask, max_iter, tol)
        (q * w).sum().backward()
        wall_fwd_bwd = time.perf_counter() - t0
        grad_err = float((x.grad.detach() - grad_ref).abs().max())
        grad_rel_err = grad_err / float(grad_ref.abs().max().clamp_min(1e-30))

        residual = res.residual.abs().amax(dim=-1)
        rows.append({
            "tol": tol,
            "max_iter": max_iter,
            "achieved_residual_max": float(residual.max()),
            "achieved_residual_mean": float(residual.mean()),
            "iterations": int(res.iterations),
            "converged_fraction": float(res.converged.float().mean()),
            "nonconvergence_rate": float((~res.converged).float().mean()),
            "grad_max_abs_error_vs_tight_reference": grad_err,
            "grad_max_rel_error_vs_tight_reference": grad_rel_err,
            "wall_clock_forward_s": wall_forward,
            "wall_clock_fwd_bwd_s": wall_fwd_bwd,
            "q_max_abs_diff_vs_reference": float((res.q - ref.q).abs().max()),
        })
    return {
        "m": m, "B": B, "vocab": VSIZE, "reference_tol": ref_tol,
        "reference_max_iter": max(max_iter, 500),
        "rows": rows,
    }


def iteration_cap_sweep(m: int, B: int, tol: float, max_iters, seed: int) -> dict:
    logits, A, b, mask = make_case(B, m, seed)
    rows = []
    for max_iter in max_iters:
        t0 = time.perf_counter()
        res = project(logits, A, b, mask, max_iter=max_iter, tol=tol)
        wall = time.perf_counter() - t0
        residual = res.residual.abs().amax(dim=-1)
        rows.append({
            "max_iter": max_iter,
            "tol": tol,
            "achieved_residual_max": float(residual.max()),
            "converged_fraction": float(res.converged.float().mean()),
            "nonconvergence_rate": float((~res.converged).float().mean()),
            "iterations_used": int(res.iterations),
            "wall_clock_forward_s": wall,
        })
    return {"m": m, "B": B, "vocab": VSIZE, "tol": tol, "rows": rows}


def boundary_probe(m: int, B: int, seed: int, collinear: bool = False,
                    fracs=(0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                           3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 0.0, -1e-4, -1e-2)) -> dict:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, VSIZE, generator=g, dtype=torch.float64)
    mask = EL_MASK.unsqueeze(0).expand(B, VSIZE).clone()
    rows_A = [MASS_ROW]
    if m == 2:
        if collinear:
            noise = torch.randn(VSIZE, generator=g, dtype=torch.float64) * 1e-4
            rows_A.append(MASS_ROW * 1.0003 + noise * EL_MASK.to(torch.float64))
        else:
            rows_A.append(VALENCE_ROW)
    A_single = torch.stack(rows_A)
    A = A_single.unsqueeze(0).expand(B, m, VSIZE).clone()

    ranges = [attainable_range(row, EL_MASK) for row in rows_A]

    results = []
    for side in ("lo", "hi"):
        for frac in fracs:
            targets = []
            for lo, hi in ranges:
                span = hi - lo
                targets.append(lo + frac * span if side == "lo" else hi - frac * span)
            b = torch.tensor(targets, dtype=torch.float64).unsqueeze(0).expand(B, m).clone()

            res = project(logits, A, b, mask, max_iter=200, tol=1e-8)
            q = res.q
            Aq = torch.einsum("bmv,bv->bm", A, q)

            solo_converged = []
            if m > 1:
                for j in range(m):
                    rj = project(logits, A[:, j:j + 1, :], b[:, j:j + 1], mask,
                                 max_iter=200, tol=1e-8)
                    aj = torch.einsum("bmv,bv->bm", A[:, j:j + 1, :], rj.q)
                    ok = rj.converged & ((aj - b[:, j:j + 1]).abs().amax(dim=-1) < 1e-6)
                    solo_converged.append(ok)
            each_solo_feasible = (
                torch.stack(solo_converged).all(dim=0) if solo_converged
                else torch.ones(B, dtype=torch.bool)
            )

            conds, multiplier_norms = [], []
            for i in range(B):
                Ai, qi = A[i], q[i]
                mu = Ai @ qi
                cov = (Ai * qi) @ Ai.T - torch.outer(mu, mu)
                conds.append(float(torch.linalg.cond(cov)) if m > 1 else float(cov.item()))
                multiplier_norms.append(float(res.lam[i].abs().max()))

            residual = (Aq - b).abs().amax(dim=-1)
            not_converged = ~res.converged
            joint_infeasible_but_each_solo_feasible = bool(
                (not_converged & each_solo_feasible).float().mean() > 0
            ) if m > 1 else False
            results.append({
                "side": side, "frac_from_boundary": frac,
                "targets": targets,
                "converged_fraction": float(res.converged.float().mean()),
                "residual_max": float(residual.max()),
                "residual_mean": float(residual.mean()),
                "multiplier_absmax_over_batch": max(multiplier_norms),
                "multiplier_absmax_median_over_batch": float(np.median(multiplier_norms)),
                "hessian_metric_max_over_batch": max(conds),
                "hessian_metric_median_over_batch": float(np.median(conds)),
                "hessian_metric_is_condition_number": m > 1,
                "iterations": int(res.iterations),
                "each_constraint_solo_feasible_fraction": (
                    float(each_solo_feasible.float().mean()) if m > 1 else None
                ),
                "fraction_joint_infeasible_but_each_solo_feasible": (
                    float((not_converged & each_solo_feasible).float().mean()) if m > 1 else None
                ),
                "joint_infeasible_but_each_solo_feasible_observed": joint_infeasible_but_each_solo_feasible,
            })
    return {
        "m": m, "B": B, "vocab": VSIZE, "collinear": collinear,
        "attainable_ranges": ranges,
        "results": results,
    }


def main(smoke: bool = False) -> Path:
    torch.set_default_dtype(torch.float64)
    B = 32 if smoke else 128
    tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
    if smoke:
        tols = [1e-4, 1e-8, 1e-12]
    max_iters = [3, 5, 8, 12, 20, 30, 50, 100] if not smoke else [3, 10, 30]

    tol_sweep_m1 = tolerance_sweep(1, B, tols, max_iter=100, seed=10, ref_tol=1e-14)
    tol_sweep_m2 = tolerance_sweep(2, B, tols, max_iter=100, seed=20, ref_tol=1e-14)
    iter_sweep_m1 = iteration_cap_sweep(1, B, tol=1e-10, max_iters=max_iters, seed=11)
    iter_sweep_m2 = iteration_cap_sweep(2, B, tol=1e-10, max_iters=max_iters, seed=21)

    boundary_m1 = boundary_probe(1, B, seed=30)
    boundary_m2_realistic = boundary_probe(2, B, seed=31, collinear=False)
    boundary_m2_collinear = boundary_probe(2, B, seed=32, collinear=True)

    worst_grad_rel_err_by_tol = {}
    for row in tol_sweep_m1["rows"] + tol_sweep_m2["rows"]:
        k = f"{row['tol']:.0e}"
        v = row["grad_max_rel_error_vs_tight_reference"]
        worst_grad_rel_err_by_tol[k] = max(worst_grad_rel_err_by_tol.get(k, 0.0), v)

    m1_lo_worst_cond = max(r["multiplier_absmax_over_batch"] for r in boundary_m1["results"] if r["side"] == "lo")
    m1_hi_worst_cond = max(r["multiplier_absmax_over_batch"] for r in boundary_m1["results"] if r["side"] == "hi")
    collinear_worst_cond = max(r["hessian_metric_max_over_batch"] for r in boundary_m2_collinear["results"])
    realistic_worst_cond = max(r["hessian_metric_max_over_batch"] for r in boundary_m2_realistic["results"])

    payload = {
        "tolerance_sweep": {"m1": tol_sweep_m1, "m2": tol_sweep_m2},
        "iteration_cap_sweep": {"m1": iter_sweep_m1, "m2": iter_sweep_m2},
        "boundary_conditioning": {
            "m1": boundary_m1,
            "m2_realistic_pair": boundary_m2_realistic,
            "m2_near_collinear_pair": boundary_m2_collinear,
        },
        "summary": {
            "worst_grad_rel_err_by_tol": worst_grad_rel_err_by_tol,
            "m1_multiplier_absmax_at_lo_boundary": m1_lo_worst_cond,
            "m1_multiplier_absmax_at_hi_boundary": m1_hi_worst_cond,
            "m2_realistic_pair_worst_condition_number": realistic_worst_cond,
            "m2_near_collinear_pair_worst_condition_number": collinear_worst_cond,
            "m2_realistic_pair_correlation_note": (
                "mass and valence rows are the same pair characterized in "
                "second_channel.py (correlation 0.725 across MP-20 training "
                "structures); this artifact does not recompute that correlation, "
                "only the conditioning of the resulting dual Hessian"
            ),
        },
    }
    return emit(
        "solver_robustness" + ("_smoke" if smoke else ""),
        payload,
        config={"vocab": VSIZE, "batch": B, "tols": tols, "max_iters": max_iters, "smoke": smoke},
        seeds=[10, 11, 20, 21, 30, 31, 32],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
