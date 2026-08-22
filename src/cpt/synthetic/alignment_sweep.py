"""Data-constraint alignment sweep: a controlled probe of multiplier decay (Risk 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cpt.common.artifacts import emit
from cpt.synthetic.targets import (
    conditioning_target,
    kl,
    tilting_target,
    tv,
    violation_rate,
)
from cpt.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "src/cpt/synthetic/alignment_sweep.py"


def sweep_point(bias: float, base_config: TaskConfig) -> dict:
    """Exact constrained-target quantities at one data-alignment level."""
    config = TaskConfig(**{**base_config.as_dict(), "markov_bias": bias})
    task = enumerate_task(config)

    base = conditioning_target(task, apply_S=True)
    data_mean = float(base @ task.total_charge)
    target = float(config.charge_target)

    rec = {
        "markov_bias": bias,
        "data_mean_charge": data_mean,
        "mean_gap": data_mean - target,
        "abs_mean_gap": abs(data_mean - target),
    }

    try:
        tilt, lam = tilting_target(task, apply_S=True, charge_target=target)
    except ValueError as exc:
        rec |= {"feasible": False, "reason": str(exc)}
        return rec

    hard = conditioning_target(task, apply_S=True, charge_equals=int(target))
    rec |= {
        "feasible": True,
        "multiplier_lambda": lam,
        "abs_multiplier": abs(lam),
        "kl_tilt_from_base_nats": kl(tilt, base),
        "kl_hard_from_base_nats": kl(hard, base),
        "tv_tilt_to_base": tv(tilt, base),
        "tv_tilt_to_hard": tv(tilt, hard),
        "tilt_violation_rate": violation_rate(task, tilt),
        "base_violation_rate": violation_rate(task, base),
        "tilt_entropy_nats": float(-np.sum(tilt[tilt > 0] * np.log(tilt[tilt > 0]))),
        "base_entropy_nats": float(-np.sum(base[base > 0] * np.log(base[base > 0]))),
    }
    return rec


def main(smoke: bool = False) -> Path:
    base_config = TaskConfig(length=4, max_count=2) if smoke else TaskConfig()
    biases = (
        [0.8, 1.0, 1.3]
        if smoke
        else [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
    )

    points = [sweep_point(b, base_config) for b in biases]
    feasible = [p for p in points if p.get("feasible")]

    gaps = np.array([p["abs_mean_gap"] for p in feasible])
    lams = np.array([p["abs_multiplier"] for p in feasible])
    kls = np.array([p["kl_tilt_from_base_nats"] for p in feasible])
    signed_gaps = np.array([p["mean_gap"] for p in feasible])
    signed_lams = np.array([p["multiplier_lambda"] for p in feasible])
    viols = np.array([p["tilt_violation_rate"] for p in feasible])

    order = np.argsort(gaps)
    near = order[: max(3, len(order) // 2)]
    fits = {}
    for label, y in (("multiplier", lams), ("kl", kls)):
        g, v = gaps[near], y[near]
        keep = (g > 1e-9) & (v > 1e-12)
        if keep.sum() >= 3:
            slope, intercept = np.polyfit(np.log(g[keep]), np.log(v[keep]), 1)
            fits[f"{label}_log_log_slope"] = float(slope)
            fits[f"{label}_log_log_intercept"] = float(intercept)

    aligned = min(feasible, key=lambda p: p["abs_mean_gap"])

    payload = {
        "sweep": points,
        "closest_to_alignment": {
            "markov_bias": aligned["markov_bias"],
            "abs_mean_gap": aligned["abs_mean_gap"],
            "abs_multiplier": aligned["abs_multiplier"],
            "kl_tilt_from_base_nats": aligned["kl_tilt_from_base_nats"],
        },
        "scaling_fits": fits,
        "monotonicity": {
            "signed_multiplier_decreases_with_signed_gap": bool(
                np.all(np.diff(signed_lams[np.argsort(signed_gaps)]) <= 1e-9)
            ),
            "multiplier_opposes_gap_sign": bool(
                np.all(signed_lams * signed_gaps <= 1e-12)
            ),
            "kl_increases_with_abs_gap_per_side": {
                side: bool(np.all(np.diff(k[np.argsort(g)]) >= -1e-9))
                for side, g, k in (
                    ("below_target", -signed_gaps[signed_gaps < 0], kls[signed_gaps < 0]),
                    ("above_target", signed_gaps[signed_gaps > 0], kls[signed_gaps > 0]),
                )
            },
        },
        "violation_decoupling": {
            "tilt_violation_min": float(viols.min()),
            "tilt_violation_max": float(viols.max()),
            "tilt_violation_range": float(viols.max() - viols.min()),
            "multiplier_fold_range": float(lams.max() / lams.min()),
            "kl_fold_range": float(kls.max() / kls.min()),
        },
        "decay_interpretation": {
            "note": (
                "At zero mean gap the multiplier is zero and the constrained target "
                "equals the base: projected's converged endpoint provably equals mask-only's. The "
                "measured scaling exponents say how fast that degeneracy is approached "
                "as data-constraint alignment improves."
            )
        },
    }

    return emit(
        "alignment_sweep" + ("_smoke" if smoke else ""),
        payload,
        config=base_config.as_dict() | {"biases": biases, "smoke": smoke},
        seeds=[base_config.markov_seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    print(main(smoke=args.smoke))
