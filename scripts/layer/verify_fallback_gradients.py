from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.layer.projection import ProjectionLayer, project, project_differentiable

SCRIPT = "scripts/layer/verify_fallback_gradients.py"

REPO_ROOT = Path(__file__).resolve().parents[2]

TRAINING_CALL_SITES = {
    "train_crystal.py:801": ("scripts/train/train_crystal.py", 801),
    "train_crystal.py:802": ("scripts/train/train_crystal.py", 802),
    "train_crystal.py:826": ("scripts/train/train_crystal.py", 826),
    "train_a2.py:332": ("scripts/train/train_a2.py", 332),
}

INFERENCE_ONLY_CALL_SITES = {
    "train_crystal.py:458": ("scripts/train/train_crystal.py", 458),
    "train_a2.py:161": ("scripts/train/train_a2.py", 161),
}


def _enclosing_function(path: Path, lineno: int) -> tuple[str, bool]:
    tree = ast.parse(path.read_text())
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = max(
                (n.lineno for n in ast.walk(node) if hasattr(n, "lineno")),
                default=start,
            )
            if start <= lineno <= end:
                if best is None or start > best.lineno:
                    best = node
    if best is None:
        raise ValueError(f"no enclosing function for {path}:{lineno}")
    no_grad = any(
        (isinstance(d, ast.Attribute) and d.attr == "no_grad")
        or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "no_grad")
        for d in best.decorator_list
    )
    return best.name, no_grad


def audit_call_sites() -> dict:
    audit = {}
    for label, (relpath, lineno) in {**TRAINING_CALL_SITES, **INFERENCE_ONLY_CALL_SITES}.items():
        path = REPO_ROOT / relpath
        lines = path.read_text().splitlines()
        source_line = lines[lineno - 1].strip()
        assert "project(" in source_line, f"{label}: expected project() call, got {source_line!r}"
        assert "project_differentiable(" not in source_line
        fn_name, no_grad_decorated = _enclosing_function(path, lineno)
        audit[label] = {
            "file": relpath,
            "line": lineno,
            "source": source_line,
            "enclosing_function": fn_name,
            "no_grad_decorated": no_grad_decorated,
            "role": "inference_only" if label in INFERENCE_ONLY_CALL_SITES else "training_loop",
        }
    return audit


def _one_discrepancy_trial(logits: torch.Tensor, A: torch.Tensor, b: torch.Tensor, dtype) -> dict:
    logits1 = logits.clone().requires_grad_(True)
    res = project(logits1, A, b, mask=None, max_iter=50, tol=1e-10)
    loss = (res.q * torch.arange(logits.shape[-1], dtype=dtype).view(1, -1)).sum()
    loss.backward()
    grad_via_plain_call = logits1.grad.clone()
    has_grad_fn = res.q.grad_fn is not None
    reaches_logits = logits1.grad is not None and torch.isfinite(logits1.grad).all() and logits1.grad.abs().sum().item() > 0

    logits2 = logits.clone().requires_grad_(True)
    q2 = project_differentiable(logits2, A, b, None, max_iter=50, tol=1e-10)
    loss2 = (q2 * torch.arange(logits.shape[-1], dtype=dtype).view(1, -1)).sum()
    loss2.backward()
    grad_via_layer = logits2.grad.clone()

    rel_diff = (grad_via_plain_call - grad_via_layer).norm().item() / max(grad_via_layer.norm().item(), 1e-30)
    return {
        "newton_iterations": res.iterations,
        "unrolled_call_has_grad_fn": has_grad_fn,
        "unrolled_call_grad_reaches_logits": bool(reaches_logits),
        "unrolled_grad_norm": grad_via_plain_call.norm().item(),
        "implicit_grad_norm": grad_via_layer.norm().item(),
        "unrolled_vs_implicit_relative_l2_diff": rel_diff,
        "unrolled_vs_implicit_max_abs_diff": (grad_via_plain_call - grad_via_layer).abs().max().item(),
    }


def confirm_autograd_flows_through_unrolled_solve(vocab: int, batch: int, m: int, seed: int, dtype=torch.float32) -> dict:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, vocab, generator=g, dtype=dtype)
    A = torch.randn(batch, m, vocab, generator=g, dtype=dtype)
    p = torch.softmax(logits, dim=-1)
    b_exact = torch.einsum("bmv,bv->bm", A, p)
    b_perturbed = b_exact + 0.1 * torch.randn(batch, m, generator=g, dtype=dtype)

    realistic = _one_discrepancy_trial(logits, A, b_perturbed, dtype)
    zero_iteration_edge_case = _one_discrepancy_trial(logits, A, b_exact, dtype)

    return {
        "vocab": vocab,
        "batch": batch,
        "m": m,
        "dtype": str(dtype),
        "realistic_perturbed_target": realistic,
        "zero_iteration_edge_case_target_equals_current_expectation": zero_iteration_edge_case,
    }


def masked_softmax_grad_analytic(logits: torch.Tensor, mask: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    neg_inf_logits = logits.masked_fill(~mask, float("-inf"))
    q = torch.softmax(neg_inf_logits, dim=-1)
    gq = grad_out * q
    grad = gq - q * gq.sum(-1, keepdim=True)
    return grad.masked_fill(~mask, 0.0)


def verify_fallback_semantics(vocab: int, m: int, seed: int) -> dict:
    torch.manual_seed(seed)
    B = 8
    logits = torch.randn(B, vocab, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(B, vocab, dtype=torch.bool)
    mask[:, vocab // 2 :] = False

    A = torch.randn(B, m, vocab, dtype=torch.float64)
    A = A * mask.unsqueeze(1)

    p_masked = torch.softmax(logits.detach().masked_fill(~mask, float("-inf")), dim=-1)
    attainable_lo = torch.where(mask, A[:, 0, :], torch.full_like(A[:, 0, :], float("inf"))).min(-1).values
    attainable_hi = torch.where(mask, A[:, 0, :], torch.full_like(A[:, 0, :], float("-inf"))).max(-1).values

    feasible_rows = torch.zeros(B, dtype=torch.bool)
    feasible_rows[: B // 2] = True
    b0 = torch.einsum("bmv,bv->bm", A, p_masked)
    b = b0.clone()
    b[~feasible_rows, 0] = attainable_hi[~feasible_rows] + 5.0

    res = project(logits, A, b, mask, max_iter=50, tol=1e-10)
    fallback_used = ~(res.converged & feasible_rows)

    q_softmax = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    q_softmax = q_softmax / q_softmax.sum(-1, keepdim=True)
    q = torch.where(fallback_used.unsqueeze(-1), q_softmax, res.q)

    grad_out = torch.randn(B, vocab, dtype=torch.float64)
    loss = (q * grad_out).sum()
    loss.backward()
    grad_logits = logits.grad.clone()

    analytic_fallback_grad = masked_softmax_grad_analytic(logits.detach(), mask, grad_out)

    fallback_idx = fallback_used.nonzero(as_tuple=True)[0].tolist()
    feasible_idx = (~fallback_used).nonzero(as_tuple=True)[0].tolist()

    per_row_fallback_diff = []
    for i in fallback_idx:
        num = (grad_logits[i] - analytic_fallback_grad[i]).abs().max().item()
        den = analytic_fallback_grad[i].abs().max().clamp_min(1e-30).item()
        per_row_fallback_diff.append({"row": i, "max_abs_diff": num, "max_abs_diff_relative": num / den})

    logits_detached_test = logits.detach().clone().requires_grad_(True)
    b_bool = b.detach().clone()
    A_bool = A.detach().clone()
    res_bool_test = project(logits_detached_test, A_bool, b_bool, mask, max_iter=50, tol=1e-10)
    bool_is_leaf_bool_dtype = (res_bool_test.converged.dtype == torch.bool) and (not res_bool_test.converged.requires_grad)

    m_multiplier = A.shape[1]
    lam_perturbed = res.lam.detach().clone()
    lam_perturbed[fallback_idx] = lam_perturbed[fallback_idx] * 1000.0 + 7.0
    q_from_perturbed_lam_at_fallback = q.detach().clone()
    output_independent_of_lambda_at_fallback = torch.allclose(
        q.detach()[fallback_idx], q_softmax.detach()[fallback_idx], atol=1e-12
    )

    return {
        "vocab": vocab,
        "m": m_multiplier,
        "batch": B,
        "n_fallback_rows": len(fallback_idx),
        "n_feasible_rows": len(feasible_idx),
        "fallback_boolean_is_plain_bool_tensor_not_differentiable": bool_is_leaf_bool_dtype,
        "per_row_fallback_vs_analytic_masked_softmax_grad": per_row_fallback_diff,
        "max_relative_diff_over_fallback_rows": max(
            (r["max_abs_diff_relative"] for r in per_row_fallback_diff), default=0.0
        ),
        "diverged_lambda_at_fallback_rows_abs_sum": [res.lam[i].abs().sum().item() for i in fallback_idx],
        "output_at_fallback_rows_equals_masked_softmax_exactly": output_independent_of_lambda_at_fallback,
    }


def main(seed: int = 0) -> Path:
    audit = audit_call_sites()
    crystal_check = confirm_autograd_flows_through_unrolled_solve(vocab=500, batch=32, m=1, seed=seed, dtype=torch.float32)
    synthetic_check = confirm_autograd_flows_through_unrolled_solve(vocab=4, batch=32, m=1, seed=seed, dtype=torch.float32)
    crystal_check_f64 = confirm_autograd_flows_through_unrolled_solve(vocab=500, batch=32, m=1, seed=seed, dtype=torch.float64)
    synthetic_check_f64 = confirm_autograd_flows_through_unrolled_solve(vocab=4, batch=32, m=1, seed=seed, dtype=torch.float64)
    fallback_crystal = verify_fallback_semantics(vocab=500, m=1, seed=seed)
    fallback_synthetic = verify_fallback_semantics(vocab=4, m=1, seed=seed)

    payload = {
        "call_site_audit": audit,
        "training_arms_use_unrolled_project_not_project_differentiable": all(
            not audit[k]["no_grad_decorated"] for k in TRAINING_CALL_SITES
        ),
        "unrolled_vs_implicit_gradient_discrepancy": {
            "crystal_arm_vocab500_float32": crystal_check,
            "synthetic_arm_vocab4_float32": synthetic_check,
            "crystal_arm_vocab500_float64": crystal_check_f64,
            "synthetic_arm_vocab4_float64": synthetic_check_f64,
        },
        "fallback_gradient_semantics": {
            "crystal_arm_vocab500": fallback_crystal,
            "synthetic_arm_vocab4": fallback_synthetic,
        },
    }

    config = {"seed": seed}
    return emit(
        "fallback_gradients",
        payload,
        config=config,
        seeds=[seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = main(seed=args.seed)
    print(path)
