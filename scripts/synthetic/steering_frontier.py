"""Steering frontier: what per-step projection buys, and what it costs."""

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
        exact_set = np.isclose(phi, c)
        accept = float(base[exact_set].sum())

        row = {
            "charge_target": int(c),
            "steering_strength": abs(base_mean - c),
            "rejection_acceptance_rate": accept,
            "rejection_draws_per_sample": (1.0 / accept) if accept > 0 else None,
            "target_reachable_by_rejection": accept > 0,
        }

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
            "induced_mean_charge": float(induced @ phi),
            "induced_mean_error": float(induced @ phi) - c,
            "induced_mass_exactly_on_target": float(induced[exact_set].sum()),
            "tilt_mass_exactly_on_target": (float(tilt[exact_set].sum()) if tilt_ok else None),
            "base_mass_exactly_on_target": accept,
            "induced_regime_S_mass": float(induced[task.feasible_S].sum()),
            "efficiency_ratio_vs_rejection": (1.0 / accept) if accept > 0 else None,
            "target_mass_enrichment": (
                float(induced[exact_set].sum()) / accept if accept > 0 else None
            ),
            "per_step_infeasible_fraction": diag["per_step_infeasible_fraction"],
        }
        COMPARABLE_TOL = 0.5
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
            "waste_fraction_of_induced_divergence": (
                (k_ind - k_ach) / k_ind if ach_ok and k_ind > 0 else None
            ),
        }
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
    if smoke:
        targets = [t for t in attainable if abs(t) <= 4][:5]
    else:
        targets = [t for t in attainable if -8 <= t <= 6 and t % 2 == 0]

    rows = frontier(task, targets)

    reachable = [r for r in rows if r["target_reachable_by_rejection"]]
    with_tilt = [r for r in reachable if r["tilt_feasible"]]
    comparable = [r for r in with_tilt if r["excess_comparison_valid"]]
    with_ach = [r for r in with_tilt
                if r["excess_kl_over_tilt_at_achieved_nats"] is not None]

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
        "corr_strength_vs_efficiency": float(np.corrcoef(strengths, np.log(ratios))[0, 1]),
        "corr_strength_vs_excess_kl": float(np.corrcoef(c_strengths, excess)[0, 1]),
        "degenerate_target_present": bool(np.any(strengths < 1.0)),
    }

    waste = np.array([r["waste_fraction_of_induced_divergence"] for r in with_ach])
    tilt_kls = np.array([r["tilt_at_achieved_kl_from_base_nats"] for r in with_ach])
    signed = np.array([r["signed_steering_strength"] for r in with_ach])

    def monotone_per_side(x: np.ndarray, y: np.ndarray) -> dict:
        """Monotonicity checked separately either side of the base mean."""
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
            "the projected arm is compared against it on efficiency, divergence, and mass on "
            "target — never on efficiency alone."
        ),
    }

    return emit(
        "steering_frontier" + ("_smoke" if smoke else ""),
        payload,
        config=cfg.as_dict() | {"targets": targets, "smoke": smoke},
        seeds=[cfg.markov_seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
