"""Verification for the batched projection layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.layer.projection import project, project_differentiable

torch.set_default_dtype(torch.float64)


def _case(B=4, V=12, m=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, V, generator=g, dtype=torch.float64)
    A = torch.randn(B, m, V, generator=g, dtype=torch.float64)
    p = torch.softmax(logits, dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p) + 0.2 * torch.randn(B, m, generator=g, dtype=torch.float64)
    return logits, A, b


def test_output_is_a_distribution():
    logits, A, b = _case()
    q = project(logits, A, b).q
    assert torch.allclose(q.sum(-1), torch.ones(q.shape[0]), atol=1e-12)
    assert bool((q >= 0).all())


def test_constraint_is_satisfied():
    logits, A, b = _case(m=2, seed=3)
    res = project(logits, A, b)
    assert bool(res.converged.all())
    assert float(res.residual.abs().max()) < 1e-9


def test_solution_has_the_exponential_tilt_form():
    """KKT: q must equal p tilted by lambda'a, up to normalization."""
    logits, A, b = _case(seed=5)
    res = project(logits, A, b)
    log_p = torch.log_softmax(logits, dim=-1)
    tilt = torch.einsum("bm,bmv->bv", res.lam, A)
    implied = torch.log_softmax(log_p + tilt, dim=-1)
    assert torch.allclose(res.q.log(), implied, atol=1e-10)


def test_is_the_kl_minimizer_among_feasible_perturbations():
    """No feasible neighbour of q has smaller KL to p. Variational check."""
    logits, A, b = _case(B=1, V=10, m=1, seed=7)
    res = project(logits, A, b)
    p = torch.softmax(logits, dim=-1)[0]
    q = res.q[0]
    a = A[0, 0]

    def kl(x):
        return float((x * (x.log() - p.log())).sum())

    base = kl(q)
    g = torch.Generator().manual_seed(11)
    for _ in range(200):
        d = torch.randn(q.shape, generator=g, dtype=torch.float64) * q
        M = torch.stack([torch.ones_like(a), a])
        coef = torch.linalg.lstsq(M.T, d.unsqueeze(-1)).solution.squeeze(-1)
        d = d - M.T @ coef
        for s in (0.02, 0.06):
            cand = q + s * d
            if bool((cand <= 0).any()):
                continue
            cand = cand / cand.sum()
            if abs(float(cand @ a) - float(b[0, 0])) > 1e-9:
                continue
            assert kl(cand) >= base - 1e-12


def test_masked_tokens_receive_zero_mass():
    logits, A, b = _case(seed=13)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, :3] = False
    p = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p)
    res = project(logits, A, b, mask)
    assert float(res.q[~mask].abs().max()) == 0.0
    assert torch.allclose(res.q.sum(-1), torch.ones(res.q.shape[0]), atol=1e-12)


def test_zero_multiplier_when_constraint_already_holds():
    """If p already satisfies the constraint, the projection is the identity."""
    logits, A, _ = _case(seed=17)
    p = torch.softmax(logits, dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p)
    res = project(logits, A, b)
    assert float(res.lam.abs().max()) < 1e-8
    assert torch.allclose(res.q, p, atol=1e-10)


@pytest.mark.parametrize("m", [1, 2])
def test_gradients_match_finite_differences(m):
    logits, A, b = _case(B=3, V=10, m=m, seed=19 + m)
    logits = logits.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x: project_differentiable(x, A, b), (logits,), eps=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("offset", [1e-6, 1e-8, 1e-9, 1e-10, 1e-12])
def test_no_residual_floor_when_near_optimal_at_initialization(offset):
    """Constructed to start where the dual is flattest — the trap configuration."""
    logits, A, _ = _case(B=4, V=16, m=1, seed=31)
    p = torch.softmax(logits, dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p) + offset

    res = project(logits, A, b)
    assert float(res.residual.abs().max()) < 1e-10, (
        f"residual floor at offset {offset:g}: {float(res.residual.abs().max()):.3e}"
    )
    assert bool(res.converged.all())


def test_residual_does_not_improve_with_more_iterations_when_converged():
    """A stalled solver looks converged only until the budget is raised."""
    logits, A, _ = _case(B=4, V=16, m=1, seed=37)
    p = torch.softmax(logits, dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p) + 1e-9

    short = project(logits, A, b, max_iter=50)
    long = project(logits, A, b, max_iter=500)
    assert float(short.residual.abs().max()) < 1e-10
    assert torch.allclose(short.q, long.q, atol=1e-14)
    assert short.iterations < 20


def test_converges_without_stalling_near_the_optimum():
    """Regression: the dual is flat to machine precision at the optimum."""
    for seed in (1161, 1323, 23, 29):
        logits, A, b = _case(B=4, V=16, m=1, seed=seed)
        res = project(logits, A, b)
        assert float(res.residual.abs().max()) < 1e-10, f"seed {seed}"
        assert res.iterations < 20, f"seed {seed}: {res.iterations} iterations"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32,
                                   torch.bfloat16, torch.float16])
def test_solver_converges_at_every_input_dtype(dtype):
    """The solve must not degrade with the training dtype."""
    torch.manual_seed(0)
    B, V, m = 64, 120, 1
    lo = torch.randn(B, V, dtype=torch.float64)
    A = torch.randn(B, m, V, dtype=torch.float64)
    b = torch.full((B, m), 0.3, dtype=torch.float64)
    mask = torch.ones(B, V, dtype=torch.bool)

    r = project(lo.to(dtype), A.to(dtype), b.to(dtype), mask)
    assert bool(r.converged.all()), (
        f"{dtype}: only {int(r.converged.sum())}/{B} converged"
    )
    assert r.q.dtype == dtype, "output dtype must match the input"
    assert torch.isfinite(r.q).all()
    q = r.q.double()
    resid = ((A.double() @ q.unsqueeze(-1)).squeeze(-1) - b.double()).abs().max()
    floor = {torch.float64: 1e-9, torch.float32: 1e-6,
             torch.float16: 1e-3, torch.bfloat16: 1e-2}[dtype]
    assert resid < floor, f"{dtype}: residual {resid:.2e} exceeds its floor {floor:.0e}"


def test_probabilities_remain_normalized_in_low_precision():
    """A cast that broke normalization would corrupt downstream sampling."""
    torch.manual_seed(1)
    B, V, m = 32, 80, 1
    lo = torch.randn(B, V, dtype=torch.float64)
    A = torch.randn(B, m, V, dtype=torch.float64)
    b = torch.zeros(B, m, dtype=torch.float64)
    mask = torch.ones(B, V, dtype=torch.bool)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 5e-2),
                       (torch.float16, 1e-2)):
        q = project(lo.to(dtype), A.to(dtype), b.to(dtype), mask).q.double()
        assert (q >= 0).all(), f"{dtype}: negative probability"
        assert (q.sum(-1) - 1.0).abs().max() < tol, f"{dtype}: not normalized"


def test_fast_hessian_path_reaches_the_same_optimum():
    """The float32-Hessian fast path must not trade accuracy for speed."""
    torch.manual_seed(0)
    B, V, m = 32, 256, 2
    lo = torch.randn(B, V, dtype=torch.float64)
    A = torch.randn(B, m, V, dtype=torch.float64)
    mask = torch.ones(B, V, dtype=torch.bool)
    b = torch.einsum("bmv,bv->bm", A, torch.softmax(lo, -1)) + 0.5

    slow = project(lo, A, b, mask, tol=1e-10)
    fast = project(lo, A, b, mask, tol=1e-10, fast_hessian_dtype=torch.float32)

    assert bool(slow.converged.all()) and bool(fast.converged.all())
    assert fast.residual.abs().max() < 1e-9, "fast path must still satisfy the constraint"
    assert (fast.q - slow.q).abs().max() < 1e-11, (
        f"fast path reached a different point: {(fast.q - slow.q).abs().max():.2e}"
    )


def test_default_hessian_precision_is_the_accurate_one():
    """A fast default would silently break the gradient path."""
    import inspect

    sig = inspect.signature(project)
    assert sig.parameters["fast_hessian_dtype"].default is None, (
        "fast_hessian_dtype must default to None (accurate float64 Hessian)"
    )
