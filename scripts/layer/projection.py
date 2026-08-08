"""Batched expectation-constraint projection layer (WS-1 / CP2).

Given per-step logits and a set of linear expectation constraints on the
next-token distribution, return the distribution closest to the model's own in
KL divergence that satisfies those constraints.

The problem, per token:

    minimize_q   KL(q || p)                with p = softmax(logits)
    subject to   A q = b,  1'q = 1,  q >= 0
                 q_i = 0 for masked i

The KL-optimal solution has the exponential-tilt form

    q_i  proportional to  p_i * exp(lambda' a_i)

where lambda is the multiplier vector for the equality constraints. So the layer
never solves a general convex program: it solves for the multipliers, which is a
smooth, low-dimensional, strictly convex root-finding problem in exactly as many
unknowns as there are active constraints — typically one or two. That is what
makes a closed-form per-step projection affordable at all, and it is the
efficiency claim CP2 measures.

Dual objective. Substituting the tilt form into the Lagrangian gives the dual

    g(lambda) = log sum_i p_i exp(lambda' a_i)  -  lambda' b

which is convex and smooth in lambda, with

    grad g = E_q[a] - b
    hess g = Cov_q[a]

so a damped Newton iteration converges quadratically near the solution. The
gradient is exactly the constraint residual, which makes the convergence test
and the constraint check the same quantity.

Backward pass. Rather than differentiating through the Newton iterations, the
layer applies the implicit function theorem at the converged multiplier: at
optimality grad g(lambda*) = 0, so the sensitivity of lambda* to the logits is
governed by Cov_q[a] rather than by the solve path. The backward cost is O(m^3)
in the number of constraints m (1-2 here), independent of how many Newton steps
the forward took, and gives the exact gradient of the true optimum rather than
the gradient of an unrolled approximation. `verify_gradients.py` checks this
against cvxpylayers, which differentiates the general cone program, and against
finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProjectionResult:
    """Output of a projection, with everything CP2 needs to report."""

    q: torch.Tensor            # (B, V) projected distribution
    lam: torch.Tensor          # (B, m) converged multipliers
    residual: torch.Tensor     # (B, m) A q - b at convergence
    iterations: int
    converged: torch.Tensor    # (B,) bool


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """log p with masked entries at -inf, computed stably."""
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def _dual(log_p: torch.Tensor, A: torch.Tensor, b: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Dual objective g(lambda) = logsumexp(log p + lambda'a) - lambda'b."""
    tilt = torch.einsum("bm,bmv->bv", lam, A)
    return torch.logsumexp(log_p + tilt, dim=-1) - (lam * b).sum(-1)


def project(
    logits: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    max_iter: int = 50,
    tol: float = 1e-10,
    damping: float = 1e-10,
    fast_hessian_dtype: torch.dtype | None = None,
) -> ProjectionResult:
    """Project a batch of next-token distributions onto A q = b.

    Args:
        logits: (B, V) unnormalized model scores.
        A:      (B, m, V) constraint matrix; row j is the statistic a_j over the vocab.
        b:      (B, m) constraint targets.
        mask:   (B, V) bool, True where a token is allowed. Masked tokens get zero mass.

    Entries that fail to converge are flagged rather than silently returned: a
    non-converged projection is not the KL-optimal distribution and must not be
    reported as one.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    B, V = logits.shape
    m = A.shape[1]
    if A.shape != (B, m, V):
        raise ValueError(f"A must be (B, m, V) = {(B, m, V)}, got {tuple(A.shape)}")
    if b.shape != (B, m):
        raise ValueError(f"b must be (B, m) = {(B, m)}, got {tuple(b.shape)}")

    # The dual Newton solve runs in float64 regardless of the input dtype, and the
    # result is cast back at the end.
    #
    # This is not defensive padding. Measured on a 64x120 batch with one
    # constraint: at float32 only 11 of 64 entries reached the 1e-10 tolerance,
    # and float16/bfloat16 raise outright because torch has no half-precision LU
    # on CPU. Under any mixed-precision training run the layer would therefore
    # either crash or silently return distributions that are not KL-optimal —
    # and a non-optimal projection reported as optimal is exactly the failure this
    # module's convergence flag exists to prevent. Solving in float64 restores
    # 64/64 convergence at every input dtype; the only residual left is the output
    # dtype's own representation floor, which no solver can beat.
    solve_dtype = torch.float64
    # Hessian precision. Defaults to float64, i.e. the accurate setting.
    #
    # float32 is 27x faster for this contraction on consumer Blackwell and does
    # not change the residual, but it DOES move the converged solution by ~1e-9:
    # Newton stops where the residual falls under tol, and a perturbed Newton
    # direction stops at a slightly different point inside that tolerance ball.
    # That is enough to make the layer non-smooth at the scale a finite-difference
    # gradient check probes, and the implicit-function-theorem backward assumes q
    # is the exact solution. So the fast path is opt-in and measured, not default.
    hess_dtype = fast_hessian_dtype or solve_dtype
    in_dtype = logits.dtype
    logits = logits.to(solve_dtype)
    A = A.to(solve_dtype)
    b = b.to(solve_dtype)

    log_p = _masked_log_softmax(logits, mask)
    lam = torch.zeros(B, m, dtype=solve_dtype, device=logits.device)
    eye = torch.eye(m, dtype=solve_dtype, device=logits.device).expand(B, m, m)
    it = 0

    for it in range(1, max_iter + 1):
        tilt = torch.einsum("bm,bmv->bv", lam, A)
        log_q = log_p + tilt
        log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
        q = log_q.exp()

        Aq = torch.einsum("bmv,bv->bm", A, q)          # E_q[a]
        grad = Aq - b                                   # residual == dual gradient
        converged = grad.abs().amax(dim=-1) < tol
        if bool(converged.all()):
            break

        # Hessian = Cov_q[a] = A diag(q) A' - (Aq)(Aq)'
        #
        # This contraction is computed in `hess_dtype` (float32), not float64, and
        # that choice is measured rather than assumed. On consumer Blackwell
        # (RTX 5090, sm_120) float64 throughput is 1.61 TFLOP/s against 69.3 in
        # float32 — a 43x hardware penalty — and this three-way contraction over
        # the vocabulary is the single dominant cost of the solve: 1.66 ms in
        # float64 against 0.062 ms in float32 at m >= 2, a 27x difference.
        #
        # Demoting it is sound because the Hessian only steers the Newton
        # DIRECTION; it does not enter the answer. A relative error of ~1.9e-7
        # changes the step slightly and so the convergence path, but the
        # fixed point is defined by the residual, which stays in float64. The
        # residual contraction is kept at full precision at no cost — measured at
        # 1.00x the float32 time, because it is a two-way reduction that is
        # bandwidth-bound rather than arithmetic-bound.
        Hc = torch.einsum(
            "bmv,bv,bnv->bmn", A.to(hess_dtype), q.to(hess_dtype), A.to(hess_dtype)
        ).to(solve_dtype)
        H = Hc - Aq.unsqueeze(-1) * Aq.unsqueeze(-2)
        H = H + damping * eye
        step = torch.linalg.solve(H, grad.unsqueeze(-1)).squeeze(-1)

        # Backtracking on the dual keeps the iteration global: a full Newton step
        # can overshoot when the constraint is far from satisfied.
        #
        # The acceptance test needs a tolerance, not a strict inequality. Near
        # the optimum the dual is flat to machine precision — a step that
        # reduces the residual by orders of magnitude changes g by less than one
        # ulp, so a strict `g1 > g0` test reads rounding noise as failure and
        # backtracks to a zero step. That stalls Newton one step short of
        # convergence and pins the residual at ~1e-9 no matter how many
        # iterations are allowed. Accepting any step that does not increase the
        # dual by more than a relative epsilon fixes it.
        slack = 1e-12 * g0_abs if (g0_abs := _dual(log_p, A, b, lam).abs()).numel() else 0.0
        t = torch.ones(B, dtype=solve_dtype, device=logits.device)
        g0 = _dual(log_p, A, b, lam)
        for _ in range(30):
            g1 = _dual(log_p, A, b, lam - t.unsqueeze(-1) * step)
            bad = (g1 > g0 + slack.clamp_min(1e-14)) & ~converged
            if not bool(bad.any()):
                break
            t = torch.where(bad, t * 0.5, t)
        lam = torch.where(converged.unsqueeze(-1), lam, lam - t.unsqueeze(-1) * step)

    tilt = torch.einsum("bm,bmv->bv", lam, A)
    log_q = log_p + tilt
    log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
    q = log_q.exp()
    residual = torch.einsum("bmv,bv->bm", A, q) - b

    # Convergence is judged on the float64 solve, before the output cast. Casting
    # first would conflate two distinct failures: a solver that did not reach the
    # optimum, and a solver that did but whose answer cannot be represented in the
    # requested dtype. Only the first is a solver failure; the second is a floor.
    converged = residual.abs().amax(dim=-1) < max(tol, 1e-8)

    return ProjectionResult(
        q=q.to(in_dtype), lam=lam.to(in_dtype),
        residual=residual.to(in_dtype), iterations=it,
        converged=converged,
    )


class ProjectionLayer(torch.autograd.Function):
    """Projection with implicit-function-theorem gradients."""

    @staticmethod
    def forward(ctx, logits, A, b, mask, max_iter, tol):
        with torch.no_grad():
            res = project(logits, A, b, mask, max_iter=max_iter, tol=tol)
        ctx.save_for_backward(res.q, A)
        ctx.mask = mask
        return res.q

    @staticmethod
    def backward(ctx, grad_out):
        q, A = ctx.saved_tensors
        B, m, V = A.shape

        Aq = torch.einsum("bmv,bv->bm", A, q)
        A_centered = A - Aq.unsqueeze(-1)
        cov = torch.einsum("bmv,bv,bnv->bmn", A_centered, q, A_centered)
        cov = cov + 1e-10 * torch.eye(m, dtype=q.dtype, device=q.device).expand(B, m, m)

        # Unconstrained softmax Jacobian term.
        gq = grad_out * q
        direct = gq - q * gq.sum(-1, keepdim=True)

        # Implicit correction: how lambda must move to hold A q = b as logits change.
        w = torch.einsum("bmv,bv->bm", A_centered, gq)
        adj = torch.linalg.solve(cov, w.unsqueeze(-1)).squeeze(-1)
        correction = torch.einsum("bm,bmv->bv", adj, A_centered) * q

        grad_logits = direct - correction
        if ctx.mask is not None:
            grad_logits = grad_logits.masked_fill(~ctx.mask, 0.0)
        return grad_logits, None, None, None, None, None


def project_differentiable(
    logits: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Projection usable inside a training graph."""
    return ProjectionLayer.apply(logits, A, b, mask, max_iter, tol)
