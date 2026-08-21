"""Seed-averaged crystal steering frontier for the manuscript's Figure 2.

Aggregation convention: SEED-AVERAGED per target, computed directly from the
18 per-run rows in results/steering_frontier_rows.json. This is the
convention reports/steering_frontier_crystal_notes.md uses for its headline
frontier table (as opposed to the per-run, 18-row convention used only for
the compensation regression in that note) -- see the note's "The frontier"
table, which this script reproduces exactly.

results/steering_frontier.json is the synthetic-task (charge) steering
sweep, not a density measurement -- its rows carry `charge_target` /
`induced_mean_charge` on the enumerable 4-symbol task, not crystal density.
It is not merged into this figure's x/y axis because the two are different
physical quantities on different scales; it is not needed to reproduce any
number plotted here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/compute_fig2_steering_frontier.py"
REPO = Path(__file__).resolve().parents[2]


def main() -> Path:
    import json

    with open(REPO / "results" / "steering_frontier_rows.json") as fh:
        src = json.load(fh)
    rows = list(src["payload"]["rows"].values())
    targets = sorted(set(r["target"] for r in rows))

    agg = []
    for t in targets:
        sel = [r for r in rows if r["target"] == t]
        dep = np.array([r["deployed_density"] for r in sel], dtype=float)
        agg.append({
            "target": float(t),
            "n_seeds": len(sel),
            "deployed_mean": float(dep.mean()),
            "deployed_sd": float(dep.std(ddof=1)) if len(sel) > 1 else 0.0,
        })

    ceiling_row = max(agg, key=lambda r: r["target"])

    payload = {
        "aggregation": "seed-averaged per target (matches reports/steering_frontier_crystal_notes.md 'The frontier' table)",
        "source_artifact": "results/steering_frontier_rows.json",
        "n_source_runs": len(rows),
        "frontier": agg,
        "steering_ceiling": {
            "target": ceiling_row["target"],
            "achieved_density": ceiling_row["deployed_mean"],
            "achieved_sd": ceiling_row["deployed_sd"],
            "definition": "achieved density at the highest requested target, where the arm saturates",
        },
        "data_mean_density": float(src["payload"]["rows"][next(iter(src["payload"]["rows"]))]["data_mean"]),
    }

    return emit(
        "fig_steering_frontier",
        payload,
        config={"source": "steering_frontier_rows"},
        seeds=[0, 1],
        script=SCRIPT,
    )


if __name__ == "__main__":
    main()
