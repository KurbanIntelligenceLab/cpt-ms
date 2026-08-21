"""Compute and emit the exact constrained targets for the synthetic task.

Produces `results/synthetic_ground_truth.json`, the artifact every synthetic
distortion result is measured against. Run with `--smoke` for the pre-flight
shard (short sequences, seconds) required before any real launch.

Usage:
    python -m scripts.synthetic.compute_ground_truth
    python -m scripts.synthetic.compute_ground_truth --smoke
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

SCRIPT = "scripts/synthetic/compute_ground_truth.py"


def summarize(task, name: str, p: np.ndarray, *, extra: dict | None = None) -> dict:
    """Summary statistics for one distribution over the enumerated sequences."""
    phi = task.total_charge.astype(np.float64)
    mean = float(p @ phi)
    rec = {
        "name": name,
        "support_size": int((p > 0).sum()),
        "entropy_nats": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
        "mean_charge": mean,
        "var_charge": float(p @ (phi - mean) ** 2),
        "hard_violation_rate": violation_rate(task, p),
        "regime_S_satisfied_mass": float(p[task.feasible_S].sum()),
        "kl_from_data_nats": kl(p, task.p_data),
        "kl_to_data_nats": kl(task.p_data, p),
        "tv_to_data": tv(p, task.p_data),
    }
    if extra:
        rec.update(extra)
    return rec


def main(smoke: bool = False) -> Path:
    config = (
        TaskConfig(length=4, max_count=2)
        if smoke
        else TaskConfig()
    )
    task = enumerate_task(config)

    p_data = task.p_data

    # Target (i): data distribution restricted to the feasible set.
    t_S = conditioning_target(task, apply_S=True, charge_equals=None)
    t_SE_hard = conditioning_target(
        task, apply_S=True, charge_equals=int(config.charge_target)
    )

    # Regime-E sequence-level ideal: the KL-optimal expectation-constrained tilt.
    t_E_tilt, lam_E = tilting_target(task, apply_S=False)
    t_SE_tilt, lam_SE = tilting_target(task, apply_S=True)

    distributions = {
        "data": summarize(task, "data", p_data),
        "S_conditioned": summarize(task, "S_conditioned", t_S),
        "SE_hard_conditioned": summarize(task, "SE_hard_conditioned", t_SE_hard),
        "E_tilt": summarize(task, "E_tilt", t_E_tilt, extra={"multiplier_lambda": lam_E}),
        "SE_tilt": summarize(task, "SE_tilt", t_SE_tilt, extra={"multiplier_lambda": lam_SE}),
    }

    # The comparison that decides which target a distortion claim is made against.
    separation = {
        "kl_tilt_vs_hard_nats": kl(t_SE_tilt, t_SE_hard),
        "kl_hard_vs_tilt_nats": kl(t_SE_hard, t_SE_tilt),
        "tv_tilt_vs_hard": tv(t_SE_tilt, t_SE_hard),
        "tilt_hard_violation_rate": violation_rate(task, t_SE_tilt),
        "hard_target_violation_rate": violation_rate(task, t_SE_hard),
    }

    payload = {
        "task": {
            "n_sequences": int(task.sequences.shape[0]),
            "vocab_size": config.vocab_size,
            "length": config.length,
            "regime_S_feasible_count": int(task.feasible_S.sum()),
            "regime_S_feasible_fraction": float(task.feasible_S.mean()),
            "data_mass_on_S_feasible": float(p_data[task.feasible_S].sum()),
            "attainable_charge_range": [
                int(task.total_charge.min()),
                int(task.total_charge.max()),
            ],
        },
        "constraint_binds": {
            "data_mean_charge": float(p_data @ task.total_charge),
            "charge_target": float(config.charge_target),
            "binding": bool(
                abs(float(p_data @ task.total_charge) - config.charge_target) > 1e-6
            ),
        },
        "distributions": distributions,
        "target_separation": separation,
    }

    return emit(
        "synthetic_ground_truth" + ("_smoke" if smoke else ""),
        payload,
        config=config.as_dict() | {"smoke": smoke},
        seeds=[config.markov_seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny shard for pre-flight")
    args = ap.parse_args()
    path = main(smoke=args.smoke)
    print(path)
