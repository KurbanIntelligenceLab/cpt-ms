"""Batched expectation-constraint projection layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProjectionResult:
    """Output of a projection, with everything needs to report."""

    q: torch.Tensor
    lam: torch.Tensor
    residual: torch.Tensor
    iterations: int
    converged: torch.Tensor


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def _dual(log_p: torch.Tensor, A: torch.Tensor, b: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
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
    """Project a batch of next-token distributions onto A q = b."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    B, V = logits.shape
    m = A.shape[1]
    if A.shape != (B, m, V):
        raise ValueError(f"A must be (B, m, V) = {(B, m, V)}, got {tuple(A.shape)}")
    if b.shape != (B, m):
        raise ValueError(f"b must be (B, m) = {(B, m)}, got {tuple(b.shape)}")

    solve_dtype = torch.float64
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

        Aq = torch.einsum("bmv,bv->bm", A, q)
        grad = Aq - b
        converged = grad.abs().amax(dim=-1) < tol
        if bool(converged.all()):
            break

        Hc = torch.einsum(
            "bmv,bv,bnv->bmn", A.to(hess_dtype), q.to(hess_dtype), A.to(hess_dtype)
        ).to(solve_dtype)
        H = Hc - Aq.unsqueeze(-1) * Aq.unsqueeze(-2)
        H = H + damping * eye
        step = torch.linalg.solve(H, grad.unsqueeze(-1)).squeeze(-1)

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

        gq = grad_out * q
        direct = gq - q * gq.sum(-1, keepdim=True)

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
