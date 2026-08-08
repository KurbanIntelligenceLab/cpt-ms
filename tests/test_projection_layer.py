"""Verification for the batched projection layer.

Properties are checked against independent characterizations of the optimum —
the KKT/tilt form, the variational definition of the I-projection, and autograd's
own finite differences — rather than against the implementation's own output.
"""

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
    """KKT: q must equal p tilted by lambda'a, up to normalization.

    Checked as a ratio identity rather than by re-deriving q, so it tests the
    optimality characterization instead of the solve path.
    """
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
        # Project the perturbation to preserve normalization and E[a].
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
    # Retarget to stay feasible on the reduced support.
    p = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p)
    res = project(logits, A, b, mask)
    assert float(res.q[~mask].abs().max()) == 0.0
    assert torch.allclose(res.q.sum(-1), torch.ones(res.q.shape[0]), atol=1e-12)


def test_zero_multiplier_when_constraint_already_holds():
    """If p already satisfies the constraint, the projection is the identity.

    This is the degenerate case the collapse test is built around, so it must be
    exact rather than approximate.
    """
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
    """Constructed to start where the dual is flattest — the trap configuration.

    The solver initializes at lambda = 0, so setting b just off the base
    distribution's own statistic puts the very first iterate within `offset` of
    the optimum. That is exactly where the dual objective is flat to machine
    precision and where a strict-decrease line search backtracks to a zero step,
    freezing the residual near 1e-9 no matter how many iterations are allowed.

    A floor there biases the active-multiplier fraction downward, which is the
    quantity CP0 gates on and which selects the paper's register — a numerically
    induced low reading would look like early internalization. Any future change
    to the line search must keep this test passing.
    """
    logits, A, _ = _case(B=4, V=16, m=1, seed=31)
    p = torch.softmax(logits, dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p) + offset

    res = project(logits, A, b)
    assert float(res.residual.abs().max()) < 1e-10, (
        f"residual floor at offset {offset:g}: {float(res.residual.abs().max()):.3e}"
    )
    assert bool(res.converged.all())


def test_residual_does_not_improve_with_more_iterations_when_converged():
    """A stalled solver looks converged only until the budget is raised.

    If the residual is limited by a line-search stall rather than by precision,
    granting more iterations changes nothing — that was the signature of the
    original defect. Here a tenfold larger budget must leave the answer alone,
    which distinguishes genuine convergence from a frozen iterate.
    """
    logits, A, _ = _case(B=4, V=16, m=1, seed=37)
    p = torch.softmax(logits, dim=-1)
    b = torch.einsum("bmv,bv->bm", A, p) + 1e-9

    short = project(logits, A, b, max_iter=50)
    long = project(logits, A, b, max_iter=500)
    assert float(short.residual.abs().max()) < 1e-10
    assert torch.allclose(short.q, long.q, atol=1e-14)
    assert short.iterations < 20


def test_converges_without_stalling_near_the_optimum():
    """Regression: the dual is flat to machine precision at the optimum.

    A strict decrease test in the line search reads rounding noise as failure and
    backtracks to a zero step, stalling Newton one step short and pinning the
    residual near 1e-9 regardless of the iteration budget. This asserts both the
    tight residual and the small iteration count, since the stall showed up as
    hitting the cap.
    """
    for seed in (1161, 1323, 23, 29):
        logits, A, b = _case(B=4, V=16, m=1, seed=seed)
        res = project(logits, A, b)
        assert float(res.residual.abs().max()) < 1e-10, f"seed {seed}"
        assert res.iterations < 20, f"seed {seed}: {res.iterations} iterations"
