"""Accuracy-diversity Pareto data for the manuscript's Figure 3.

Aggregated over SUB-CEILING targets only (target <= the steering ceiling
identified in results/FIG2_steering_frontier.json, 10.405 g/cm^3): above the
ceiling the projection's deployed density saturates regardless of the
request, so a comparison against alternative methods there measures the
ceiling rather than the methods.

Sources:
  results/WS6_best_of_n.json          -- best-of-n curve, n in {1,4,16,64};
                                          error = mean_best_error, diversity =
                                          unique_formula_fraction, each
                                          averaged over sub-ceiling targets.
  results/WS3_conditional_baseline.json -- the A3 (projection) point: this
                                          artifact's four targets (3.0, 5.0,
                                          8.0, 10.0) are already inside the
                                          sub-ceiling range, and are its own
                                          per_target.<t>.A3 rows (err, unq),
                                          averaged.

GAP: results/WS6_smc_steering.json carries no diversity / unique-formula
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
    with open(REPO / "results" / "FIG2_steering_frontier.json") as fh:
        fig2 = json.load(fh)["payload"]
    ceiling = fig2["steering_ceiling"]["achieved_density"]

    with open(REPO / "results" / "WS6_best_of_n.json") as fh:
        bon = json.load(fh)["payload"]
    with open(REPO / "results" / "WS3_conditional_baseline.json") as fh:
        cb = json.load(fh)["payload"]
    with open(REPO / "results" / "WS6_smc_steering.json") as fh:
        smc = json.load(fh)["payload"]

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

    payload = {
        "steering_ceiling_g_per_cm3": ceiling,
        "sub_ceiling_targets": sub_ceiling_targets,
        "best_of_n_curve": bon_curve,
        "projection_point": {
            "mean_error": float(a3_err),
            "mean_unique_fraction": float(a3_unq),
            "targets_averaged": cb_sub,
            "source": "results/WS3_conditional_baseline.json per_target.<t>.A3",
        },
        "gap": {
            "smc_diversity_unavailable": (
                "results/WS6_smc_steering.json rows carry only mean_error, "
                "sd_error, wall_clock_s_per_output, forward_passes_per_output "
                "-- no unique-formula-fraction or any other diversity field "
                "for any (target, K) cell, so SMC cannot be placed on this "
                "figure's diversity axis without fabricating a number. Its "
                "error-only comparison is reported in reports/"
                "WS6b_smc_notes.md instead."
            )
        },
    }

    return emit(
        "FIG3_accuracy_diversity_pareto",
        payload,
        config={"ceiling_source": "FIG2_steering_frontier"},
        seeds=[0, 1],
        script=SCRIPT,
    )


if __name__ == "__main__":
    main()
