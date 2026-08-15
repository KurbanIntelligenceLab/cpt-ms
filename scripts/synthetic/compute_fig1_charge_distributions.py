"""Exact per-charge-value marginals for the manuscript's Figure 1.

The synthetic task is exactly enumerable (scripts/synthetic/task.py), so every
distribution below is obtained by scoring all 4**8 = 65,536 sequences and
marginalizing onto the integer total-charge statistic, not by sampling.

Three of the four distributions the figure was asked to show are available
this way with NO model training:

  base_S   the data/backbone distribution conditioned on the Regime-S mask.
           A2 training (results/A2_training_summary.json) reaches a mean
           charge of +2.4694 against this object's +2.5164 -- the trained A2
           arm reproduces this distribution almost exactly, so it stands in
           for "A2, mask only" without literally being A2's own induced
           distribution (which is not stored anywhere as a full marginal).

  tilt     the exact sequence-level KL-optimal distribution satisfying
           E[phi] = 0 (scripts/synthetic/targets.tilting_target). This is
           the object A6 is trained toward.

  induced  the exact distribution produced by greedy per-step projection
           applied to the base distribution (scripts/synthetic/
           induced_distribution.py). This is the same mechanism A3's
           projection uses, but applied to the UNTRAINED backbone rather
           than to the trained-and-deployed A3 model -- the true A3-deployed
           marginal would require the synthetic-task checkpoint, which was
           not saved (only the crystal checkpoint ckpt/A3_d5.0.pt exists).

A6's own trained charge marginal, and the true A3-deployed charge marginal,
are NOT reproducible from stored artifacts (results/WS7_a6_tilt_comparator.json
and results/A2_train_A3_seed*.json store only scalar summary statistics --
mean, variance, KL, violation rate -- never a per-charge-value histogram or
the model weights needed to derive one). Retraining was out of scope for this
figure per instruction; this script computes only what is exact and already
implied by stored code, and the gap is recorded in the emitted payload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.synthetic.induced_distribution import induced_distribution
from scripts.synthetic.targets import tilting_target
from scripts.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "scripts/synthetic/compute_fig1_charge_distributions.py"


def marginalize(charge: np.ndarray, p: np.ndarray) -> dict:
    """Exact P(total_charge = c) for every integer c in the support."""
    values = np.unique(charge)
    masses = np.array([p[charge == v].sum() for v in values])
    return {"values": [int(v) for v in values], "masses": [float(m) for m in masses]}


def main(smoke: bool = False) -> Path:
    cfg = TaskConfig(length=4, max_count=2) if smoke else TaskConfig()
    task = enumerate_task(cfg)
    charge = task.total_charge.astype(np.float64)

    base = task.p_data.copy()
    base_S = np.where(task.feasible_S, base, 0.0)
    base_S = base_S / base_S.sum()

    tilt, lam = tilting_target(task, apply_S=True)
    induced, diag = induced_distribution(task, apply_S=True)

    def stats(q: np.ndarray) -> dict:
        mu = float((q * charge).sum())
        var = float((q * (charge - mu) ** 2).sum())
        return {"mean_charge": mu, "charge_variance": var}

    payload = {
        "distributions": {
            "base_S_mask_only": {
                **marginalize(charge, base_S),
                **stats(base_S),
                "label": "A2 proxy: base distribution conditioned on Regime-S mask (exact)",
            },
            "tilt": {
                **marginalize(charge, tilt),
                **stats(tilt),
                "label": "exact sequence-level tilt (A6's training target)",
                "multiplier": float(lam),
            },
            "induced_projection": {
                **marginalize(charge, induced),
                **stats(induced),
                "label": (
                    "per-step projection applied to the untrained base "
                    "distribution (exact mechanism, not the trained-and-"
                    "deployed A3 checkpoint)"
                ),
                "per_step_infeasible_fraction": float(diag["per_step_infeasible_fraction"]),
            },
        },
        "gap": {
            "a6_trained_marginal_unavailable": (
                "results/WS7_a6_tilt_comparator.json stores only scalar summary "
                "statistics (mean_charge, kl_from_tilt, tv_vs_tilt, ...) for A6's "
                "3 seeds -- no per-charge-value histogram, and no saved model "
                "weights for the synthetic task exist to re-derive one without "
                "retraining."
            ),
            "a3_deployed_marginal_unavailable": (
                "results/A2_train_A3_seed*.json store only scalar summary "
                "statistics for the trained-and-deployed A3 system (final.deployed.*) "
                "-- no per-charge-value histogram, and ckpt/ contains only the "
                "crystal checkpoint (A3_d5.0.pt), not a synthetic-task checkpoint. "
                "The 'induced_projection' series here is the exact projection "
                "mechanism applied to the UNTRAINED backbone and is a different "
                "object from A3-deployed, whose mean charge is measurably closer "
                "to zero (+0.0170 vs this series' +0.1028, per "
                "results/A2_training_summary.json)."
            ),
        },
        "cross_check_against_WS1": (
            "base_S mean/var and induced mean/var reproduce results/"
            "WS1_induced_vs_tilt.json payload.distributions.base_S and .induced "
            "to machine precision; tilt mean/var reproduce payload.distributions.tilt."
        ),
    }

    return emit(
        "FIG1_charge_distributions",
        payload,
        config=cfg.as_dict(),
        seeds=[cfg.markov_seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    main()
