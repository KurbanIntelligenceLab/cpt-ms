from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.common.arms import pick
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/compute_infeasibility_correlation.py"
REPO = Path(__file__).resolve().parents[2]


def main() -> Path:
    with open(REPO / "results" / "steering_frontier_rows.json") as fh:
        frontier_src = json.load(fh)
    rows = list(frontier_src["payload"]["rows"].values())
    targets = sorted(set(r["target"] for r in rows))

    with open(REPO / "results" / "conditional_baseline.json") as fh:
        cond_src = json.load(fh)
    cond_per_target = dict(cond_src["payload"]["per_target"])

    per_target = []
    for t in targets:
        sel = [r for r in rows if r["target"] == t]
        init = np.array([r["infeas_init"] for r in sel], dtype=float)
        final = np.array([r["infeas"] for r in sel], dtype=float)
        deployed = np.array([r["deployed_density"] for r in sel], dtype=float)
        abs_error = float(abs(deployed.mean() - t))
        key = f"{t:.1f}"
        unq = pick(cond_per_target.get(key, {}), "projected").get("unq") if cond_per_target.get(key) else None
        per_target.append({
            "target": float(t),
            "n_seeds": len(sel),
            "infeas_init_mean": float(init.mean()),
            "infeas_final_mean": float(final.mean()),
            "repair_factor_ratio_of_means": float(init.mean() / final.mean()),
            "deployed_density_mean": float(deployed.mean()),
            "abs_error_vs_target": abs_error,
            "unique_formula_fraction_projected": (float(unq) if unq is not None else None),
        })

    tgt_arr = np.array([p["target"] for p in per_target])
    infeas_init_arr = np.array([p["infeas_init_mean"] for p in per_target])
    infeas_final_arr = np.array([p["infeas_final_mean"] for p in per_target])
    error_arr = np.array([p["abs_error_vs_target"] for p in per_target])

    def corr_block(x, y):
        n = len(x)
        sp = stats.spearmanr(x, y)
        pe = stats.pearsonr(x, y)
        return {
            "n": int(n),
            "spearman_r": float(sp.correlation),
            "spearman_p": float(sp.pvalue),
            "pearson_r": float(pe.statistic),
            "pearson_p": float(pe.pvalue),
        }

    error_vs_infeas_final = corr_block(infeas_final_arr, error_arr)
    error_vs_infeas_init = corr_block(infeas_init_arr, error_arr)
    error_vs_target = corr_block(tgt_arr, error_arr)

    div_rows = [p for p in per_target if p["unique_formula_fraction_projected"] is not None]
    diversity_targets_available = [p["target"] for p in div_rows]
    if len(div_rows) >= 3:
        div_infeas = np.array([p["infeas_final_mean"] for p in div_rows])
        div_unq = np.array([p["unique_formula_fraction_projected"] for p in div_rows])
        diversity_vs_infeas_final = corr_block(div_infeas, div_unq)
    else:
        diversity_vs_infeas_final = None

    payload = {
        "locus_note": (
            "infeas_init and infeas in steering_frontier_rows.json are both measured "
            "at the INFERENCE locus by scripts/train/train_crystal.py's sample() function "
            "with project_at_inference=True and use_mask=True, called once against the "
            "untrained weights at step 0 (infeas_init) and once against the trained weights "
            "at the final step (infeas, the deployed reading). Both are the per-step "
            "infeasible-step fraction averaged over generated sequences, i.e. weighted by "
            "the model's own induced sampling distribution, matching the convention "
            "results/training_summary.json payload.notes calls probability-weighted -- "
            "not the unweighted enumeration-mean reading that note contrasts it with. "
            "Neither number is a TEACHER-FORCED training-time rate: the projected training loop "
            "(scripts/train/train_crystal.py lines ~707-829) does run the projection against "
            "real MP-20 prefixes every training step, but it does not accumulate an "
            "infeasible-step counter for that pass the way sample() does for joint's "
            "joint_stats; no training-locus infeasibility series exists in the repo for the "
            "projected arm. So 'infeas_init vs infeas' answers 'before training vs after training, "
            "both read at inference', not 'training vs inference' as literally phrased by "
            "the reviewer; this artifact reports that distinction rather than fabricating "
            "the missing training-locus series."
        ),
        "diversity_note": (
            "No per-target unique-formula fraction exists for all 18 frontier runs. "
            "results/conditional_baseline.json reports the projected arm's unique_formula_fraction "
            "for exactly 4 of the 10 frontier targets (3.0, 5.0, 8.0, 10.0), 2 seeds each. Its "
            "per-seed dep_density and dep_infeas fields are bit-exact matches (16+ significant "
            "figures) to the corresponding deployed_density and infeas fields in "
            "steering_frontier_rows.json, so this is the SAME underlying set of runs read "
            "through a second extraction script, not an independent evaluation job -- the "
            "unique_formula_fraction values are additional fields recorded on the same rows, "
            "not a fresh sample. results/zero_shot_retargeting.json's on-target diagonal "
            "cells (3.0, 5.0, 10.0, single seed, checkpoint-based) give a third, non-identical "
            "reading of three of those targets from a genuinely separate job and are not "
            "merged in here. The diversity correlation below is reported at n=4 and, since the "
            "infeasibility and diversity values share the same runs rather than an independent "
            "replication, is not claimed to generalize beyond these four targets."
        ),
        "per_target": per_target,
        "diversity_targets_available": diversity_targets_available,
        "correlations": {
            "abs_error_vs_infeas_final": error_vs_infeas_final,
            "abs_error_vs_infeas_init": error_vs_infeas_init,
            "abs_error_vs_target": error_vs_target,
            "unique_formula_fraction_vs_infeas_final": diversity_vs_infeas_final,
        },
    }

    return emit(
        "infeasibility_correlation",
        payload,
        config={"source": ["steering_frontier_rows", "conditional_baseline"]},
        seeds=[0, 1],
        script=SCRIPT,
    )


if __name__ == "__main__":
    main()
