"""Infeasibility at convergence vs. requested target, and the repair factor.

Source: results/WS3_steering_frontier_rows.json (18 per-run rows, each
carrying `infeas_init` and `infeas` -- the running-budget infeasibility
fraction before and after training through the projection).

Repair factor convention: RATIO OF SEED-AVERAGED MEANS per target,
mean(infeas_init) / mean(infeas), matching reports/WS3_steering_frontier_notes.md
("29.2x at target 1.5, 0.97x at 20.0"). The alternative convention, the
per-run mean of per-run ratios, gives 29.39x at target 1.5 -- the "29.3x"
figure a prior session flagged as a rounding inconsistency with the frontier
notes, not the correct number. Both are computed here and only the
ratio-of-means is plotted, with the alternative recorded for provenance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/compute_fig4_infeasibility.py"
REPO = Path(__file__).resolve().parents[2]


def main() -> Path:
    with open(REPO / "results" / "WS3_steering_frontier_rows.json") as fh:
        src = json.load(fh)
    rows = list(src["payload"]["rows"].values())
    targets = sorted(set(r["target"] for r in rows))

    per_target = []
    for t in targets:
        sel = [r for r in rows if r["target"] == t]
        init = np.array([r["infeas_init"] for r in sel], dtype=float)
        final = np.array([r["infeas"] for r in sel], dtype=float)
        ratio_of_means = float(init.mean() / final.mean())
        mean_of_ratios = float(np.mean(init / final))
        per_target.append({
            "target": float(t),
            "n_seeds": len(sel),
            "infeas_init_mean": float(init.mean()),
            "infeas_final_mean": float(final.mean()),
            "repair_factor_ratio_of_means": ratio_of_means,
            "repair_factor_mean_of_ratios_alt": mean_of_ratios,
        })

    payload = {
        "repair_factor_convention": "ratio of seed-averaged means per target (matches reports/WS3_steering_frontier_notes.md: 29.2x at target 1.5, 0.97x at 20.0)",
        "per_target": per_target,
        "weakest_steering_repair_factor": per_target[0]["repair_factor_ratio_of_means"],
        "strongest_steering_repair_factor": per_target[-1]["repair_factor_ratio_of_means"],
    }

    return emit(
        "FIG4_infeasibility_vs_target",
        payload,
        config={"source": "WS3_steering_frontier_rows"},
        seeds=[0, 1],
        script=SCRIPT,
    )


if __name__ == "__main__":
    main()
