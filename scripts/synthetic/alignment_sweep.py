"""Data-constraint alignment sweep: a controlled probe of multiplier decay (Risk 1).

The research plan (v2) states the mechanism in risk-register item 1 — constraints
internalized at convergence turn the projection into an identity, so A3's
endpoint equals A2's — and states the expectation in its "Abstract register"
section: multiplier decay is the modal outcome rather than a tail risk, because
training data is real materials and so satisfies the constraints, and any
constraint the data satisfies will see multipliers shrink as the model learns.
On MP-20 both are claims available only after training. Here it is a dial.

`markov_bias` controls how far the data distribution's mean charge sits from the
constraint target. At bias = 1 the chain is unbiased; above 1 it favors positive
charge. Sweeping it traces multiplier magnitude as a function of
data-constraint alignment, with the exact answer known at every point.

What is measured at each alignment level, all in closed form:

  lambda*        the exact multiplier the sequence-level tilt requires. This is
                 the quantity that decays. At perfect alignment it is zero and
                 the constrained target equals the base — the degenerate case in
                 which A3's endpoint provably equals A2's.

  KL(tilt||base) the work the constraint does. Decays quadratically in the mean
                 gap near alignment, so it vanishes faster than the gap itself.

  TV, violation  the per-sample cost, which by Lemma 1 are the same number.

This is the cheapest experiment in the project and it answers the question the
whole CP0 gate is built around, so it runs before any crystal training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.synthetic.task import TaskConfig, enumerate_task
from scripts.synthetic.targets import (
    conditioning_target,
    kl,
    tilting_target,
    tv,
    violation_rate,
)

SCRIPT = "scripts/synthetic/alignment_sweep.py"


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

    # Near-alignment scaling: |lambda| ~ gap and KL ~ gap^2 are the expected
    # exponential-family rates. Fit on the closest-aligned half of the sweep.
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
        # Monotonicity is checked in the SIGNED gap, not its magnitude. The tilt
        # is not symmetric about the alignment point — the attainable charge
        # distribution is skewed — so two points at nearly equal |gap| on
        # opposite sides can order either way without contradicting anything.
        "monotonicity": {
            # lambda is strictly DECREASING in the signed gap: when the data mean
            # sits above the target the multiplier is negative, pulling mass back
            # down, and vice versa. Monotone decrease is the correctness check.
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
        # The per-sample violation rate of the tilt barely moves across the sweep
        # even as the multiplier and the KL move by orders of magnitude. Recorded
        # explicitly because it means per-sample violation is NOT a proxy for
        # constraint activity: a near-flat violation rate is compatible with the
        # constraint doing anything from almost no work to a great deal of it.
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
                "equals the base: A3's converged endpoint provably equals A2's. The "
                "measured scaling exponents say how fast that degeneracy is approached "
                "as data-constraint alignment improves."
            )
        },
    }

    return emit(
        "WS2_alignment_sweep" + ("_smoke" if smoke else ""),
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
