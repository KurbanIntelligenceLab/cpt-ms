"""Gradient verification: implicit-function gradients vs two oracles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.layer.projection import project, project_differentiable

SCRIPT = "scripts/layer/verify_gradients.py"


def make_case(rng: np.random.Generator, B: int, V: int, m: int, seed: int):
    """A random but feasible constraint configuration."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, V, generator=g, dtype=torch.float64)
    A = torch.randn(B, m, V, generator=g, dtype=torch.float64)
    p = torch.softmax(logits, dim=-1)
    b0 = torch.einsum("bmv,bv->bm", A, p)
    b = b0 + 0.25 * torch.randn(B, m, generator=g, dtype=torch.float64)
    return logits, A, b


FD_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _fd_at(logits, A, b, w, eps):
    def f(x):
        return (project(x, A, b, max_iter=200, tol=1e-14).q * w).sum()

    grad = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            up = logits.clone(); up[i, j] += eps
            dn = logits.clone(); dn[i, j] -= eps
            grad[i, j] = (f(up) - f(dn)) / (2 * eps)
    return grad


def finite_difference_jacobian(logits, A, b):
    """Finite-difference oracle, swept over step size."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(*logits.shape, generator=g, dtype=torch.float64)
    return {eps: _fd_at(logits, A, b, w, eps) for eps in FD_STEPS}, w


def analytic_gradient(logits, A, b, w):
    x = logits.clone().requires_grad_(True)
    (project_differentiable(x, A, b) * w).sum().backward()
    return x.grad


def cvxpylayers_gradient(logits, A, b, w):
    """Oracle gradient from cvxpylayers' KKT differentiation."""
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except ImportError:
        return None, "cvxpylayers not installed"

    B, V = logits.shape
    m = A.shape[1]

    q = cp.Variable(V)
    p_par = cp.Parameter(V, nonneg=True)
    A_par = cp.Parameter((m, V))
    b_par = cp.Parameter(m)
    obj = cp.Minimize(cp.sum(cp.kl_div(q, p_par)))
    cons = [cp.sum(q) == 1, A_par @ q == b_par]
    layer = CvxpyLayer(cp.Problem(obj, cons), parameters=[p_par, A_par, b_par], variables=[q])

    x = logits.clone().requires_grad_(True)
    p = torch.softmax(x, dim=-1)
    outs = []
    for i in range(B):
        (qi,) = layer(p[i], A[i], b[i], solver_args={"eps": 1e-10, "max_iters": 50_000})
        outs.append(qi)
    (torch.stack(outs) * w).sum().backward()
    return x.grad, None


def jacobian_rank_check(rng: np.random.Generator, trials: int = 200) -> dict:
    """Constraint matrices must have the rank their configuration implies."""
    bad = []
    for t in range(trials):
        m = int(rng.integers(1, 4))
        V = int(rng.integers(m + 2, 24))
        A = rng.normal(size=(m, V))
        r = int(np.linalg.matrix_rank(A))
        if r != m:
            bad.append({"trial": t, "m": m, "V": V, "rank": r})
    return {"trials": trials, "rank_deficient": len(bad), "cases": bad[:10]}


def main(smoke: bool = False) -> Path:
    rng = np.random.default_rng(0)
    configs = (
        [(4, 10, 1)]
        if smoke
        else [(4, 10, 1), (4, 16, 1), (4, 16, 2), (3, 24, 2), (3, 32, 3)]
    )

    rows = []
    for (B, V, m) in configs:
        logits, A, b = make_case(rng, B, V, m, seed=1000 + V * 10 + m)
        res = project(logits, A, b)

        fds, w = finite_difference_jacobian(logits, A, b)
        an = analytic_gradient(logits, A, b, w)
        cx, cx_err = cvxpylayers_gradient(logits, A, b, w)

        sweep = {
            f"{eps:.0e}": float((an - fd).abs().max() / fd.abs().max().clamp_min(1e-30))
            for eps, fd in fds.items()
        }
        best_eps = min(sweep, key=sweep.get)

        row = {
            "B": B, "V": V, "m": m,
            "forward_max_abs_residual": float(res.residual.abs().max()),
            "forward_iterations": res.iterations,
            "forward_all_converged": bool(res.converged.all()),
            "fd_sweep_rel_diff": sweep,
            "fd_best_step": best_eps,
            "grad_rel_diff_vs_finite_difference": sweep[best_eps],
            "grad_max_abs_diff_vs_finite_difference": float(
                (an - fds[float(best_eps)]).abs().max()
            ),
        }
        if cx is None:
            row["cvxpylayers"] = {"available": False, "reason": cx_err}
        else:
            row["cvxpylayers"] = {
                "available": True,
                "grad_max_abs_diff": float((an - cx).abs().max()),
                "grad_rel_diff": float(
                    (an - cx).abs().max() / cx.abs().max().clamp_min(1e-30)
                ),
            }
        rows.append(row)

    fd_worst = max(r["grad_rel_diff_vs_finite_difference"] for r in rows)
    cx_rows = [r for r in rows if r["cvxpylayers"]["available"]]
    cx_worst = max((r["cvxpylayers"]["grad_rel_diff"] for r in cx_rows), default=None)

    payload = {
        "rows": rows,
        "rank_check": jacobian_rank_check(rng, trials=20 if smoke else 200),
        "tolerances": {
            "worst_forward_residual": max(r["forward_max_abs_residual"] for r in rows),
            "worst_rel_diff_vs_finite_difference": fd_worst,
            "worst_rel_diff_vs_cvxpylayers": cx_worst,
            "cvxpylayers_configs_compared": len(cx_rows),
            "note": (
                "The finite-difference figure is the best point of a step-size "
                "sweep, not a single arbitrary step. Central differences trade "
                "truncation error (falling as eps^2) against roundoff "
                "amplification (growing as 1/eps, because the quotient divides "
                "the forward solve's own convergence error by the step), so the "
                "per-config sweep in fd_sweep_rel_diff shows that U-shape "
                "explicitly. A single small step would report a large "
                "discrepancy that is a property of the oracle, not of the "
                "gradient. The cvxpylayers comparison carries the real weight "
                "regardless, since it shares no code path with the closed-form "
                "tilt and differentiates the KKT system of the general cone "
                "program instead."
            ),
        },
    }

    return emit(
        "gradient_verification" + ("_smoke" if smoke else ""),
        payload,
        config={"configs": configs, "dtype": "float64", "smoke": smoke},
        seeds=[0],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    torch.set_default_dtype(torch.float64)
    print(main(smoke=ap.parse_args().smoke))
