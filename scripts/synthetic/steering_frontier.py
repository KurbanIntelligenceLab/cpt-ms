"""Steering frontier: what per-step projection buys, and what it costs (WS-1).

The positive capability claim, presented as a trade with a measured price rather
than a single ratio. A ratio at one target distance grows without bound as the
target is pushed further and reads as engineered; the frontier across steering
strengths does not.

Three legs, all exact by enumeration:

  (a) EFFICIENCY. Rejection sampling is the exactness baseline: draw from the base
      model, keep only sequences with the exact target statistic. Its expected
      cost is 1 / P_base(phi == c) draws per accepted sample, and it is exact by
      construction. A3 pays a fixed per-step cost instead. The efficiency ratio is
      how many base draws rejection sampling needs for one sample that A3 produces
      per sequence.

  (b) DIVERGENCE COST. What A3 pays for that speed, measured against the
      sequence-level tilt, which is the KL-optimal distribution meeting the same
      expectation constraint. This belongs in the result, not filed as a
      limitation: it IS the price in the trade.

  (c) QUALITY AT TARGET. The fraction of A3's own mass landing exactly on the
      target statistic, and its Regime-S compliance. Rejection sampling scores 1.0
      on the first by construction; A3 does not, and the gap is the honest cost of
      an expectation constraint versus a hard one.

Steering strength is the distance from the target to the base model's own mean, so
strength zero is the degenerate matched-alignment case where the projection has
nothing to do. Reporting the whole range makes the transition visible instead of
picking a flattering point on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.synthetic.induced_distribution import induced_distribution
from scripts.synthetic.targets import kl, tilting_target, tv
from scripts.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "scripts/synthetic/steering_frontier.py"


def frontier(task, targets: list[int]) -> list[dict]:
    """Compute all three legs at each target."""
    phi = task.total_charge.astype(np.float64)

    base = np.where(task.feasible_S, task.p_data, 0.0)
    base = base / base.sum()
    base_mean = float(base @ phi)

    rows = []
    for c in targets:
        # (a) Rejection sampling: exact, and its cost is the reciprocal of the base
        # mass on the exact-target set. An empty set means unreachable at any
        # sample budget, which is the strongest form of the efficiency argument.
        exact_set = np.isclose(phi, c)
        accept = float(base[exact_set].sum())

        row = {
            "charge_target": int(c),
            "steering_strength": abs(base_mean - c),
            "rejection_acceptance_rate": accept,
            "rejection_draws_per_sample": (1.0 / accept) if accept > 0 else None,
            "target_reachable_by_rejection": accept > 0,
        }

        # (b) and (c): the tilt is the optimum for this target; induced is what A3
        # actually produces.
        try:
            tilt, lam = tilting_target(task, apply_S=True, charge_target=float(c))
            tilt_ok = True
        except ValueError:
            tilt, lam, tilt_ok = None, None, False

        induced, diag = induced_distribution(task, apply_S=True, charge_target=float(c))

        row |= {
            "tilt_feasible": tilt_ok,
            "tilt_multiplier": (float(lam) if tilt_ok else None),
            "tilt_kl_from_base_nats": (kl(tilt, base) if tilt_ok else None),
            "induced_kl_from_base_nats": kl(induced, base),
            "tv_induced_tilt": (tv(induced, tilt) if tilt_ok else None),
            # (c) quality at target
            "induced_mean_charge": float(induced @ phi),
            "induced_mean_error": float(induced @ phi) - c,
            "induced_mass_exactly_on_target": float(induced[exact_set].sum()),
            "tilt_mass_exactly_on_target": (float(tilt[exact_set].sum()) if tilt_ok else None),
            "base_mass_exactly_on_target": accept,
            "induced_regime_S_mass": float(induced[task.feasible_S].sum()),
            # efficiency: how many base draws rejection needs per A3 sequence
            "efficiency_ratio_vs_rejection": (1.0 / accept) if accept > 0 else None,
            # concentration achieved relative to the base, on the exact-target set
            "target_mass_enrichment": (
                float(induced[exact_set].sum()) / accept if accept > 0 else None
            ),
            "per_step_infeasible_fraction": diag["per_step_infeasible_fraction"],
        }
        # Excess divergence is a COST only when both distributions solve the same
        # problem. Where the per-step surrogate fails to reach the target, it is
        # not paying the constraint the tilt pays, and the difference in
        # divergence is not a cost — it can even come out negative, which would
        # read as the surrogate beating a provable optimum. Comparability is
        # therefore gated on the induced distribution actually meeting the
        # constraint, and non-comparable cells report None rather than a number
        # that invites the wrong reading.
        COMPARABLE_TOL = 0.5  # absolute mean-charge error, in charge units
        comparable = tilt_ok and abs(row["induced_mean_error"]) <= COMPARABLE_TOL
        row["excess_comparison_valid"] = bool(comparable)
        row["excess_comparison_note"] = (
            "" if comparable else
            f"induced misses the target by {abs(row['induced_mean_error']):.3f}; "
            "it is not satisfying the constraint the tilt satisfies, so the "
            "divergence difference is not a cost"
        )
        row["excess_kl_over_tilt_at_target_nats"] = (
            row["induced_kl_from_base_nats"] - row["tilt_kl_from_base_nats"]
            if comparable else None
        )

        # The comparator that is valid at EVERY target: the tilt at the mean the
        # induced distribution actually achieves. Both distributions then satisfy
        # the same expectation constraint, so the divergence difference is a
        # genuine efficiency cost — how much more the per-step route pays to reach
        # where it got, versus the optimal distribution reaching the same place.
        # Non-negative by construction, since the tilt is optimal for its own
        # constraint; a negative value would indicate a solver defect.
        achieved = float(induced @ phi)
        try:
            tilt_ach, lam_ach = tilting_target(
                task, apply_S=True, charge_target=achieved
            )
            k_ach = kl(tilt_ach, base)
            ach_ok = True
        except ValueError:
            k_ach, lam_ach, ach_ok = None, None, False

        k_ind = row["induced_kl_from_base_nats"]
        row |= {
            "tilt_at_achieved_kl_from_base_nats": k_ach,
            "tilt_at_achieved_multiplier": (float(lam_ach) if ach_ok else None),
            "excess_kl_over_tilt_at_achieved_nats": (k_ind - k_ach) if ach_ok else None,
            # What FRACTION of A3's divergence is wasted on the per-step surrogate
            # rather than spent on the constraint. Excess is roughly a fixed
            # overhead while the necessary divergence scales with steering
            # strength, so the waste fraction falls as steering strengthens.
            "waste_fraction_of_induced_divergence": (
                (k_ind - k_ach) / k_ind if ach_ok and k_ind > 0 else None
            ),
        }
        # Signed strength, because the frontier is not symmetric about the base
        # mean: the attainable charge range is not centred there, so targets above
        # and below behave differently and a single monotonicity claim over
        # |strength| would be false.
        row["signed_steering_strength"] = float(c - base_mean)
        rows.append(row)
    return rows


def main(smoke: bool = False) -> Path:
    cfg = TaskConfig(length=4, max_count=2) if smoke else TaskConfig()
    task = enumerate_task(cfg)
    phi = task.total_charge.astype(np.float64)
    base = np.where(task.feasible_S, task.p_data, 0.0)
    base = base / base.sum()
    base_mean = float(base @ phi)

    attainable = sorted({int(x) for x in np.unique(task.total_charge[task.feasible_S])})
    # Sweep targets from the base's own mean down through zero and into the
    # opposite sign, so strength zero (degenerate) and large strength both appear.
    if smoke:
        targets = [t for t in attainable if abs(t) <= 4][:5]
    else:
        targets = [t for t in attainable if -8 <= t <= 6 and t % 2 == 0]

    rows = frontier(task, targets)

    reachable = [r for r in rows if r["target_reachable_by_rejection"]]
    with_tilt = [r for r in reachable if r["tilt_feasible"]]
    # Divergence-cost statistics use only comparable cells; efficiency
    # statistics use every reachable target, since efficiency does not depend
    # on the surrogate hitting the constraint.
    comparable = [r for r in with_tilt if r["excess_comparison_valid"]]
    # The achieved-mean comparator is valid at every target, so the divergence-
    # cost statistics use it and cover the whole frontier.
    with_ach = [r for r in with_tilt
                if r["excess_kl_over_tilt_at_achieved_nats"] is not None]

    # The headline is the frontier's SHAPE, so summarize it as a relationship
    # rather than as a single ratio.
    strengths = np.array([r["steering_strength"] for r in with_tilt])
    ratios = np.array([r["efficiency_ratio_vs_rejection"] for r in with_tilt])
    excess = np.array([r["excess_kl_over_tilt_at_achieved_nats"] for r in with_ach])
    enrich = np.array([r["target_mass_enrichment"] for r in with_tilt])
    c_strengths = np.array([r["steering_strength"] for r in with_ach])

    summary = {
        "base_mean_charge": base_mean,
        "n_targets": len(rows),
        "n_reachable_by_rejection": len(reachable),
        "n_unreachable_by_rejection": len(rows) - len(reachable),
        "efficiency_ratio_range": [float(ratios.min()), float(ratios.max())],
        "excess_kl_range_nats": [float(excess.min()), float(excess.max())],
        "target_mass_enrichment_range": [float(enrich.min()), float(enrich.max())],
        # Does the price rise with the benefit? A frontier where excess divergence
        # is flat while efficiency climbs would be too good to be true.
        "corr_strength_vs_efficiency": float(np.corrcoef(strengths, np.log(ratios))[0, 1]),
        "corr_strength_vs_excess_kl": float(np.corrcoef(c_strengths, excess)[0, 1]),
        "degenerate_target_present": bool(np.any(strengths < 1.0)),
    }

    # The headline: excess divergence is roughly a fixed overhead while the
    # NECESSARY divergence scales with steering strength, so the fraction of A3's
    # divergence that is wasted collapses as steering strengthens.
    waste = np.array([r["waste_fraction_of_induced_divergence"] for r in with_ach])
    tilt_kls = np.array([r["tilt_at_achieved_kl_from_base_nats"] for r in with_ach])
    signed = np.array([r["signed_steering_strength"] for r in with_ach])

    def monotone_per_side(x: np.ndarray, y: np.ndarray) -> dict:
        """Monotonicity checked separately either side of the base mean.

        The attainable charge range is not centred on the base mean, so the
        frontier is asymmetric and a single claim over |strength| would be false —
        the same sign asymmetry recorded in the alignment sweep.
        """
        out = {}
        for label, sel in (("below_base", x < 0), ("above_base", x > 0)):
            if sel.sum() < 2:
                out[label] = None
                continue
            order = np.argsort(np.abs(x[sel]))
            out[label] = bool(np.all(np.diff(y[sel][order]) <= 1e-9))
        return out

    summary |= {
        "waste_fraction_range": [float(waste.min()), float(waste.max())],
        "tilt_kl_range_nats": [float(tilt_kls.min()), float(tilt_kls.max())],
        "tilt_kl_dynamic_range": float(tilt_kls.max() / max(tilt_kls.min(), 1e-12)),
        "excess_kl_mean_nats": float(excess.mean()),
        "excess_kl_sd_nats": float(excess.std()),
        "corr_strength_vs_waste_fraction": float(np.corrcoef(c_strengths, waste)[0, 1]),
        "n_comparable_at_target": len(comparable),
        "n_with_achieved_comparator": len(with_ach),
        "min_excess_kl_nats": float(excess.min()),
        "all_excess_non_negative": bool((excess >= -1e-9).all()),
        "n_excluded_at_target_comparator": len(with_tilt) - len(comparable),
        "waste_fraction_falls_with_strength_per_side": monotone_per_side(signed, waste),
        "interpretation": (
            "Excess divergence over the sequence-level optimum is approximately a "
            "fixed overhead (mean {:.4f}, sd {:.4f} nats) while the divergence the "
            "optimum itself must pay spans {:.0f}x across the sweep. The wasted "
            "fraction therefore falls from {:.1%} at the weakest steering to {:.1%} "
            "at the strongest: per-step projection is almost all overhead where the "
            "base model already satisfies the constraint, and almost all necessary "
            "work where it does not."
        ).format(
            float(excess.mean()), float(excess.std()),
            float(tilt_kls.max() / max(tilt_kls.min(), 1e-12)),
            float(waste[np.argmin(c_strengths)]), float(waste[np.argmax(c_strengths)]),
        ),
    }

    payload = {
        "frontier": rows,
        "summary": summary,
        "baseline_definition": (
            "Rejection sampling from the S-conditioned base, accepting only "
            "sequences whose total charge equals the target exactly. Exact by "
            "construction; expected cost 1/acceptance draws per accepted sample. "
            "A3 is compared against it on efficiency, divergence, and mass on "
            "target — never on efficiency alone."
        ),
    }

    return emit(
        "WS1_steering_frontier" + ("_smoke" if smoke else ""),
        payload,
        config=cfg.as_dict() | {"targets": targets, "smoke": smoke},
        seeds=[cfg.markov_seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
