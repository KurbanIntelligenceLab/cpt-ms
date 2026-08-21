"""CP2 overhead benchmark: projection layer vs the mask-only masked-softmax baseline.

Measures the pre-registered quantity in `preregistration/overhead_budget.md`:
per-token wall-clock overhead of the Regime-E projection against the identical mask-only
masked-softmax forward pass, reported per active-constraint count, with a 30%
ceiling.

Design decisions that matter for the number being meaningful:

  SAME PASS, NOT A MICROBENCHMARK. The mask-only arm is the identical masked-softmax
  forward the projected arm performs before projecting, so the comparison isolates the
  projection's marginal cost rather than comparing two unrelated kernels.

  FLOAT64 INTERNAL SOLVE. The projection solves in float64 regardless of input
  dtype, because at float32 only a fraction of batch entries converge and half
  precision raises outright. That is what will run in training, so it is what is
  benchmarked; timing the input-dtype version would measure unusable code.

  CUDA SYNCHRONIZATION. Every timed region is bracketed by
  `torch.cuda.synchronize()`. Without it the CPU races ahead of the queued
  kernels and the projection appears nearly free.

  WARMUP AND MEDIAN. The first calls include allocator and autotune costs.
  Warmup iterations are discarded and the median over repeats is reported
  alongside the mean, since a single slow iteration from a background process
  skews a mean on a shared box.

The hardware is recorded in the artifact. CP2's threshold is pre-registered
against the training setting, so a benchmark from a different GPU class is not a
valid test of it.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.layer.projection import ProjectionLayer, project

SCRIPT = "scripts/layer/benchmark_overhead.py"


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The mask-only baseline: exactly the pass projected performs before projecting."""
    z = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(z, dim=-1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_region(fn, device, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    a = np.asarray(samples)
    return {
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "min_s": float(a.min()),
        "p95_s": float(np.percentile(a, 95)),
        "sd_s": float(a.std()),
        "repeats": repeats,
    }


def hardware() -> dict:
    """Recorded in the artifact: the threshold is tied to the training setting."""
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info |= {
            "gpu_name": p.name,
            "gpu_total_memory_gb": round(p.total_memory / 1024**3, 2),
            "gpu_capability": f"sm_{p.major}{p.minor}",
            "cuda_runtime": torch.version.cuda,
            "arch_list": torch.cuda.get_arch_list(),
            # A capability absent from the arch list means kernels cannot launch
            # even though is_available() reports True — the documented sm_120 trap.
            "capability_in_arch_list": f"sm_{p.major}{p.minor}" in torch.cuda.get_arch_list(),
        }
    return info


def main(smoke: bool = False) -> Path:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = hardware()
    if dev.type == "cuda" and not hw.get("capability_in_arch_list", True):
        raise RuntimeError(
            f"torch has no kernels for {hw['gpu_capability']} "
            f"(arch list {hw['arch_list']}); benchmark would measure nothing"
        )

    torch.manual_seed(0)
    B = 64 if smoke else 256
    V = 512 if smoke else 4096
    repeats = 5 if smoke else 40
    warmup = 2 if smoke else 10
    constraint_counts = (1, 2) if smoke else (1, 2, 4, 8)

    rows = []
    for m in constraint_counts:
        logits = torch.randn(B, V, device=dev, dtype=torch.float32)
        A = torch.randn(B, m, V, device=dev, dtype=torch.float32)
        mask = torch.rand(B, V, device=dev) > 0.15
        mask[:, 0] = True                      # never fully mask a row
        q0 = _masked_softmax(logits, mask)
        # Feasible targets: the base distribution's own moments, so the solve is
        # not measured on an infeasible problem it would abandon early.
        b = torch.einsum("bmv,bv->bm", A, q0).detach()

        base = _time_region(lambda: _masked_softmax(logits, mask), dev, repeats, warmup)
        proj = _time_region(
            lambda: project(logits, A, b, mask, max_iter=50, tol=1e-10),
            dev, repeats, warmup,
        )
        # Second arm: float32 Hessian. Opt-in in the layer because it perturbs the
        # stopping point, but the dominant cost on consumer Blackwell, where
        # float64 throughput is ~43x lower than float32. Reported alongside so the
        # accuracy/speed trade is visible rather than chosen silently.
        fast = _time_region(
            lambda: project(logits, A, b, mask, max_iter=50, tol=1e-10,
                            fast_hessian_dtype=torch.float32),
            dev, repeats, warmup,
        )
        res_fast = project(logits, A, b, mask, max_iter=50, tol=1e-10,
                           fast_hessian_dtype=torch.float32)

        res = project(logits, A, b, mask, max_iter=50, tol=1e-10)
        # Overhead is the MARGINAL cost: the projection call already includes its
        # own masked-softmax, so subtracting the baseline gives the added time.
        row = {
            "n_constraints": m,
            "batch": B,
            "vocab": V,
            "baseline": base,
            "projection": proj,
            "projection_fast_hessian": fast,
            "overhead_ratio_median_fast": (fast["median_s"] - base["median_s"]) / base["median_s"],
            "per_token_projection_fast_us": fast["median_s"] / B * 1e6,
            "fast_vs_slow_speedup": proj["median_s"] / fast["median_s"],
            "fast_max_residual": float(res_fast.residual.abs().max()),
            "fast_agrees_with_slow": float((res_fast.q - res.q).abs().max()),
            "overhead_ratio_mean": (proj["mean_s"] - base["mean_s"]) / base["mean_s"],
            "overhead_ratio_median": (proj["median_s"] - base["median_s"]) / base["median_s"],
            "per_token_baseline_us": base["median_s"] / B * 1e6,
            "per_token_projection_us": proj["median_s"] / B * 1e6,
            "converged_fraction": float(res.converged.float().mean()),
            "iterations": int(res.iterations),
            "max_residual": float(res.residual.abs().max()),
        }
        rows.append(row)

    # Gradient path timed separately: training pays forward AND backward, so a
    # forward-only figure would understate the training-time cost.
    grad_rows = []
    for m in constraint_counts:
        logits = torch.randn(B, V, device=dev, dtype=torch.float32, requires_grad=True)
        A = torch.randn(B, m, V, device=dev, dtype=torch.float32)
        mask = torch.rand(B, V, device=dev) > 0.15
        mask[:, 0] = True
        with torch.no_grad():
            b = torch.einsum("bmv,bv->bm", A, _masked_softmax(logits, mask)).detach()

        def base_bwd():
            q = _masked_softmax(logits, mask)
            q.sum().backward()
            logits.grad = None

        def proj_bwd():
            q = ProjectionLayer.apply(logits, A, b, mask, 50, 1e-10)
            q.sum().backward()
            logits.grad = None

        bb = _time_region(base_bwd, dev, max(repeats // 2, 3), max(warmup // 2, 1))
        pb = _time_region(proj_bwd, dev, max(repeats // 2, 3), max(warmup // 2, 1))
        grad_rows.append({
            "n_constraints": m,
            "baseline_fwd_bwd": bb,
            "projection_fwd_bwd": pb,
            "overhead_ratio_median": (pb["median_s"] - bb["median_s"]) / bb["median_s"],
        })

    fwd = [r["overhead_ratio_median"] for r in rows]
    fwd_fast = [r["overhead_ratio_median_fast"] for r in rows]
    fb = [r["overhead_ratio_median"] for r in grad_rows]
    payload = {
        "hardware": hw,
        "forward_only": rows,
        "forward_and_backward": grad_rows,
        "threshold": 0.30,
        "summary": {
            "worst_forward_overhead_ratio": max(fwd),
            "worst_forward_overhead_ratio_fast_hessian": max(fwd_fast),
            "fast_hessian_speedup_range": [min(r["fast_vs_slow_speedup"] for r in rows),
                                           max(r["fast_vs_slow_speedup"] for r in rows)],
            "worst_fast_disagreement": max(r["fast_agrees_with_slow"] for r in rows),
            "worst_fwd_bwd_overhead_ratio": max(fb),
            "forward_within_budget": bool(max(fwd) <= 0.30),
            "fwd_bwd_within_budget": bool(max(fb) <= 0.30),
            "constraint_counts": list(constraint_counts),
            "all_converged": bool(all(r["converged_fraction"] == 1.0 for r in rows)),
        },
        "notes": (
            "Overhead is marginal cost over the identical masked-softmax pass, "
            "median over repeats with CUDA synchronization around every timed "
            "region and warmup discarded. The projection solves in float64 "
            "internally, which is what runs in training."
        ),
    }
    return emit(
        "overhead_benchmark" + ("_smoke" if smoke else ""),
        payload,
        config={"batch": B, "vocab": V, "repeats": repeats, "warmup": warmup,
                "constraint_counts": list(constraint_counts), "smoke": smoke},
        seeds=[0],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
