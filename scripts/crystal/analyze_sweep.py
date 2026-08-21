"""Steering sweep analysis and the compensation regression (WS-3).

Two questions from one sweep:

  STEERING. Does deployed density track the requested target across the range, and
  how does the divergence cost behave? This is the crystal analogue of the synthetic
  steering frontier.

  COMPENSATION. The crystal backbone reads far ABOVE the data mean under a constraint
  pulling density down. The hypothesis is that the backbone biases its unaided
  predictions to offset a projection it knows will pull one way, so the post-projection
  result matches the data. If so, the backbone's offset scales with the projection's
  pull across targets.

The test is a regression of offset on pull. A slope near 1 is compensation; a flat
slope means the offset is target-independent and compensation is not the explanation.
The control cell is the target at the data median, where the projection has almost
nothing to pull against and both quantities should approach zero — a slope that
survives with that cell included is the stronger result.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/analyze_sweep.py"


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Least squares with an intercept, plus the statistics needed to judge it."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    resid = y - fit
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Standard error of the slope, so "near 1" can be judged rather than asserted.
    dof = n - 2
    s2 = ss_res / dof if dof > 0 else float("nan")
    sxx = float(((x - x.mean()) ** 2).sum())
    se_slope = float(np.sqrt(s2 / sxx)) if sxx > 0 and np.isfinite(s2) else float("nan")
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "slope_stderr": se_slope,
        "slope_t_vs_zero": float(beta[1] / se_slope) if se_slope > 0 else float("nan"),
        "slope_t_vs_one": float((beta[1] - 1.0) / se_slope) if se_slope > 0 else float("nan"),
        "r_squared": float(r2),
        "n": int(n),
        "dof": int(dof),
    }


def rejection_cost(target: float, tol: float = 0.5) -> dict:
    """Expected draws for rejection sampling from the data distribution.

    This is the efficiency baseline, computed exactly from MP-20 rather than
    estimated: the fraction of structures within tol of the target is the acceptance
    rate, and its reciprocal is the expected number of draws per accepted sample.

    It is deliberately generous to the baseline — it assumes a generator that
    reproduces the data distribution perfectly, which mask-only approximates but does not
    achieve. A projection that beats THIS is beating an idealized filter.
    """
    from scripts.crystal.mp20 import load_split
    global _DENSITY_CACHE
    try:
        d = _DENSITY_CACHE
    except NameError:
        d = np.array([c.density for c in load_split("train")])
        _DENSITY_CACHE = d
    acc = float(((d > target - tol) & (d < target + tol)).mean())
    return {
        "target": target,
        "tolerance": tol,
        "acceptance_rate": acc,
        "expected_draws": (1.0 / acc) if acc > 0 else None,
        "data_fraction_percent": acc * 100.0,
        "in_data_bulk": acc >= 0.05,
    }


def load_sweep(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        p = json.load(open(f))["payload"]
        fin = p["final"]
        dep = fin.get("deployed")
        if dep is None or fin.get("mean_density") is None or dep.get("mean_density") is None:
            continue
        rows.append({
            "target": float(p["train_config"]["density_target"]),
            "seed": int(p["train_config"]["seed"]),
            "backbone_density": float(fin["mean_density"]),
            "deployed_density": float(dep["mean_density"]),
            "projection_pull": float(fin["mean_density"] - dep["mean_density"]),
            "backbone_offset_vs_data": float(fin["mean_density"] - p["data_density"]["mean"]),
            "data_mean": float(p["data_density"]["mean"]),
            "infeasible_fraction": float(dep["per_step_infeasible_fraction"]),
            "infeasible_at_init": float(p["init_reference"]["per_step_infeasible_fraction"])
            if p.get("init_reference") else None,
            "structural_validity": float(dep["structural_validity_rate"]),
            "density_computable": float(dep["density_computable_rate"]),
            "wall_clock_s": float(p["wall_clock_s"]),
        })
    return rows


def main(pattern: str, smoke: bool = False) -> Path:
    rows = load_sweep(pattern)
    if not rows:
        raise SystemExit(f"no sweep artifacts matched {pattern!r}")

    tgt = np.array([r["target"] for r in rows])
    dep = np.array([r["deployed_density"] for r in rows])
    pull = np.array([r["projection_pull"] for r in rows])
    off = np.array([r["backbone_offset_vs_data"] for r in rows])
    data_mean = rows[0]["data_mean"]

    # Steering: does the deployed density track the request?
    steering = _ols(tgt, dep)

    # Compensation, with and without the control cell, so the control's leverage is
    # visible rather than buried. The control is the target nearest the data median.
    compensation = _ols(pull, off)
    ctrl = float(tgt[np.argmin(np.abs(tgt - np.median(np.unique(tgt))))])
    keep = np.abs(tgt - ctrl) > 1e-9
    compensation_no_ctrl = _ols(pull[keep], off[keep]) if keep.sum() >= 3 else None

    per_target = {}
    for t in sorted(set(tgt.tolist())):
        sel = [r for r in rows if abs(r["target"] - t) < 1e-9]
        per_target[f"{t:g}"] = {
            "n_seeds": len(sel),
            "deployed_density_mean": float(np.mean([r["deployed_density"] for r in sel])),
            "deployed_density_sd": float(np.std([r["deployed_density"] for r in sel], ddof=1))
            if len(sel) > 1 else 0.0,
            "abs_error": float(abs(np.mean([r["deployed_density"] for r in sel]) - t)),
            "backbone_density_mean": float(np.mean([r["backbone_density"] for r in sel])),
            "projection_pull_mean": float(np.mean([r["projection_pull"] for r in sel])),
            "backbone_offset_mean": float(np.mean([r["backbone_offset_vs_data"] for r in sel])),
            "infeasible_fraction_mean": float(np.mean([r["infeasible_fraction"] for r in sel])),
            "structural_validity_mean": float(np.mean([r["structural_validity"] for r in sel])),
        }

    verdict_compensation = (
        "compensation supported: offset scales with pull"
        if compensation["slope_t_vs_zero"] > 2.0 and compensation["slope"] > 0.3
        else "compensation NOT supported: offset does not scale with pull"
    )

    # Efficiency baseline per target, so the frontier's three legs match the
    # synthetic study: accuracy, divergence cost, and cost against rejection.
    rejection = {f"{t:g}": rejection_cost(t) for t in sorted(set(tgt.tolist()))}
    bulk = [k for k, v in rejection.items() if v["in_data_bulk"]]

    payload = {
        "n_runs": len(rows),
        "rejection_baseline": rejection,
        "efficiency_note": (
            "Rejection sampling from the data distribution costs "
            f"{min(v['expected_draws'] for v in rejection.values()):.1f} to "
            f"{max(v['expected_draws'] for v in rejection.values()):.1f} draws across "
            f"these targets; {len(bulk)} of {len(rejection)} sit in the data bulk "
            "(acceptance >= 5%), where the projection's advantage is fidelity and "
            "determinism rather than sampling efficiency. The efficiency argument "
            "requires tail targets."
        ),
        "targets": sorted(set(tgt.tolist())),
        "control_target": ctrl,
        "data_density_mean": data_mean,
        "rows": rows,
        "per_target": per_target,
        "steering_fit": steering,
        "compensation_fit": compensation,
        "compensation_fit_excluding_control": compensation_no_ctrl,
        "verdict": {
            "steering_slope": steering["slope"],
            "steering_r2": steering["r_squared"],
            "compensation_slope": compensation["slope"],
            "compensation_r2": compensation["r_squared"],
            "compensation": verdict_compensation,
            "validity_exactly_one_everywhere": bool(
                all(abs(r["structural_validity"] - 1.0) < 1e-12 for r in rows)
            ),
        },
    }
    return emit(
        "steering_sweep" + ("_smoke" if smoke else ""),
        payload,
        config={"pattern": pattern, "smoke": smoke},
        seeds=sorted({r["seed"] for r in rows}),
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/sweep_t*.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(pattern=a.pattern, smoke=a.smoke))
