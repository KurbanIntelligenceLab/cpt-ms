"""Accuracy-diversity Pareto data for the manuscript's Figure 3.

Aggregated over SUB-CEILING targets only (target <= the steering ceiling
identified in results/fig_steering_frontier.json, 10.405 g/cm^3): above the
ceiling the projection's deployed density saturates regardless of the
request, so a comparison against alternative methods there measures the
ceiling rather than the methods.

Sources:
  results/best_of_n.json          -- best-of-n curve, n in {1,4,16,64};
                                          error = mean_best_error, diversity =
                                          unique_formula_fraction, each
                                          averaged over sub-ceiling targets.
  results/conditional_baseline.json -- the A3 (projection) point: this
                                          artifact's four targets (3.0, 5.0,
                                          8.0, 10.0) are already inside the
                                          sub-ceiling range, and are its own
                                          per_target.<t>.A3 rows (err, unq),
                                          averaged.

GAP: results/smc_steering.json carries no diversity / unique-formula
field for any (target, K) cell -- only mean_error, sd_error, and cost
fields. SMC therefore cannot be placed on this figure's y-axis (unique
fraction) without fabricating a number, and is reported as a documented gap
rather than plotted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/compute_fig3_pareto.py"
REPO = Path(__file__).resolve().parents[2]


def main() -> Path:
    with open(REPO / "results" / "fig_steering_frontier.json") as fh:
        fig2 = json.load(fh)["payload"]
    ceiling = fig2["steering_ceiling"]["achieved_density"]

    with open(REPO / "results" / "best_of_n.json") as fh:
        bon = json.load(fh)["payload"]
    with open(REPO / "results" / "conditional_baseline.json") as fh:
        cb = json.load(fh)["payload"]
    with open(REPO / "results" / "smc_steering.json") as fh:
        smc = json.load(fh)["payload"]
    with open(REPO / "results" / "hybrid_selection.json") as fh:
        hyb = json.load(fh)["payload"]
    with open(REPO / "results" / "smc_diversity.json") as fh:
        smc_div = json.load(fh)["payload"]

    sub_ceiling_targets = sorted({r["target"] for r in bon["rows"] if r["target"] <= ceiling + 1e-9})

    bon_curve = []
    for n in bon["n_values"]:
        errs = [r["mean_best_error"] for r in bon["rows"] if r["n"] == n and r["target"] in sub_ceiling_targets]
        divs = [r["unique_formula_fraction"] for r in bon["diversity"]
                if r["n"] == n and r["target"] in sub_ceiling_targets]
        bon_curve.append({
            "n": n,
            "mean_error": float(np.mean(errs)),
            "mean_unique_fraction": float(np.mean(divs)),
            "n_targets_averaged": len(errs),
        })

    key_by_float = {float(k): k for k in cb["per_target"].keys()}
    cb_sub = [t for t in key_by_float if t <= ceiling + 1e-9]
    a3_err = np.mean([cb["per_target"][key_by_float[t]]["A3"]["err"] for t in cb_sub])
    a3_unq = np.mean([cb["per_target"][key_by_float[t]]["A3"]["unq"] for t in cb_sub])

    hyb_sub_targets = sorted({r["target"] for r in hyb["rows"] if r["target"] <= ceiling + 1e-9})
    hybrid_curve = []
    for n in hyb["n_values"]:
        errs = [r["mean_best_error"] for r in hyb["rows"] if r["n"] == n and r["target"] in hyb_sub_targets]
        divs = [r["unique_formula_fraction"] for r in hyb["diversity"]
                if r["n"] == n and r["target"] in hyb_sub_targets]
        hybrid_curve.append({
            "n": n,
            "mean_error": float(np.mean(errs)),
            "mean_unique_fraction": float(np.mean(divs)),
            "n_targets_averaged": len(errs),
        })

    smc_tuned_targets = [r["target"] for r in smc_div["rows"]]
    smc_tuned_sub = [t for t in smc_tuned_targets if t <= ceiling + 1e-9]
    smc_tuned_point = {
        "K": smc_div["K"],
        "sigma": smc_div["sigma"],
        "mean_error": float(np.mean([r["mean_error"] for r in smc_div["rows"] if r["target"] in smc_tuned_sub])),
        "mean_unique_fraction": float(np.mean(
            [r["unique_formula_fraction"] for r in smc_div["rows"] if r["target"] in smc_tuned_sub])),
        "targets_averaged": smc_tuned_sub,
        "source": "results/smc_diversity.json rows",
    }

    payload = {
        "steering_ceiling_g_per_cm3": ceiling,
        "sub_ceiling_targets": sub_ceiling_targets,
        "best_of_n_curve": bon_curve,
        "excluded_projection_point": {
            "mean_error": float(a3_err),
            "mean_unique_fraction": float(a3_unq),
            "targets_averaged": cb_sub,
            "source": "results/conditional_baseline.json per_target.<t>.A3",
            "why_excluded": (
                "conditional_baseline's err is |mean density - target|, the error OF THE "
                "MEAN, whereas best_of_n_curve, hybrid_curve and smc_tuned_point all report "
                "the mean of PER-SAMPLE absolute errors. Measured on one shared pool of 512 "
                "samples per target, the two statistics differ by 2.3x to 2.7x across seeds, "
                "the error of the mean being the smaller whenever the samples have spread. "
                "Plotting both on one axis overstated the projection's accuracy relative to "
                "every other arm. Projection-alone is instead read off hybrid_curve at n=1, "
                "which is the same model and the same statistic as the arms it is compared to."
            ),
        },
        "hybrid_curve": hybrid_curve,
        "hybrid_curve_note": (
            "results/hybrid_selection.json: n steered samples drawn per target from "
            "the projection-deployed A3 checkpoint, keep the closest on density. n=1 is "
            "projection-alone; n>1 adds best-of-n selection on top of the steered pool."
        ),
        "smc_tuned_point": smc_tuned_point,
        "smc_tuned_point_note": (
            "results/smc_tuned.json identifies sigma=0.5 as the best fixed potential "
            "width at K=16 over a nine-point sweep (0.1-3.0), evaluated honestly (chosen "
            "without looking at per-target error, unlike the biased best-per-target row in "
            "that artifact). results/smc_diversity.json then measures unique-formula "
            "fraction for that tuned (K=16, sigma=0.5) setting across 100 independent SMC "
            "runs per target, giving SMC its first placement on this figure's diversity axis."
        ),
    }

    return emit(
        "fig_accuracy_diversity_pareto",
        payload,
        config={"ceiling_source": "fig_steering_frontier"},
        seeds=[0, 1],
        script=SCRIPT,
    )


if __name__ == "__main__":
    main()
