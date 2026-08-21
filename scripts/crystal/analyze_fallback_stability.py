from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import RESULTS_DIR, emit, load

SCRIPT = "scripts/crystal/analyze_fallback_stability.py"


def frontier_relation() -> dict:
    record = json.loads((RESULTS_DIR / "steering_frontier_rows.json").read_text())
    rows = list(record["payload"]["rows"].values())
    targets = np.array([r["target"] for r in rows])
    infeas = np.array([r["infeas"] for r in rows])
    infeas_init = np.array([r["infeas_init"] for r in rows])
    dep_density = np.array([r["deployed_density"] for r in rows])
    dep_sd = np.array([r["deployed_sd"] for r in rows])
    err = np.abs(dep_density - targets)

    r_err, p_err = stats.pearsonr(infeas, err)
    r_sd, p_sd = stats.pearsonr(infeas, dep_sd)
    r_err_init, p_err_init = stats.pearsonr(infeas_init, err)
    r_sd_init, p_sd_init = stats.pearsonr(infeas_init, dep_sd)

    return {
        "source_artifact": "results/steering_frontier_rows.json",
        "n_runs": len(rows),
        "infeas_deployed_range": [float(infeas.min()), float(infeas.max())],
        "infeas_init_range": [float(infeas_init.min()), float(infeas_init.max())],
        "pearson_r_infeas_deployed_vs_abs_density_error": float(r_err),
        "pearson_p_infeas_deployed_vs_abs_density_error": float(p_err),
        "pearson_r_infeas_deployed_vs_deployed_sd": float(r_sd),
        "pearson_p_infeas_deployed_vs_deployed_sd": float(p_sd),
        "pearson_r_infeas_init_vs_abs_density_error": float(r_err_init),
        "pearson_p_infeas_init_vs_abs_density_error": float(p_err_init),
        "pearson_r_infeas_init_vs_deployed_sd": float(r_sd_init),
        "pearson_p_infeas_init_vs_deployed_sd": float(p_sd_init),
    }


def training_dynamics_relation() -> dict:
    crystal = load("stability_regrun")["payload"]
    synthetic = load("synthetic_stability_regrun")["payload"]

    def within_run_corrs(payload):
        out = {}
        for k, v in payload["per_target"].items():
            infeas = np.array(v["per_step_infeasible_fraction_trace"])
            grad = np.array(v["grad_norm_trace"])
            loss = np.array(v["loss_trace"])
            r_grad = float(np.corrcoef(infeas, grad)[0, 1]) if infeas.std() > 0 and grad.std() > 0 else None
            r_loss = float(np.corrcoef(infeas, loss)[0, 1]) if infeas.std() > 0 and loss.std() > 0 else None
            out[k] = {"within_run_r_infeas_vs_grad_norm": r_grad, "within_run_r_infeas_vs_loss": r_loss,
                      "mean_infeasible_fraction": v["mean_infeasible_fraction_full_run"]}
        return out

    return {
        "crystal_arm": {
            "source_artifact": "results/stability_regrun.json",
            "note": "crystal A3 re-runs at fixed density targets converge to near-zero "
                    "training-time infeasibility once trained at that target; the ceiling "
                    "regime is only visible in the frontier's deployed (zero-shot) reading, "
                    "not in matched-target training curves",
            "within_run": within_run_corrs(crystal),
        },
        "synthetic_arm": {
            "source_artifact": "results/synthetic_stability_regrun.json",
            "cross_target_pearson_r_infeas_vs_loss_variance_full_run": synthetic[
                "pearson_r_infeas_vs_loss_variance_full_run"],
            "cross_target_pearson_r_infeas_vs_grad_norm_variance_full_run": synthetic[
                "pearson_r_infeas_vs_grad_norm_variance_full_run"],
            "cross_target_pearson_r_infeas_vs_loss_variance_late_window": synthetic[
                "pearson_r_infeas_vs_loss_variance_late_window"],
            "cross_target_pearson_r_infeas_vs_grad_norm_variance_late_window": synthetic[
                "pearson_r_infeas_vs_grad_norm_variance_late_window"],
            "within_run": within_run_corrs(synthetic),
            "cross_target_summary": synthetic["cross_target_summary"],
        },
    }


def main() -> Path:
    base_record = load("fallback_gradients")
    payload = dict(base_record["payload"])
    payload["training_stability_relation"] = {
        "deployed_outcome_vs_infeasibility": frontier_relation(),
        "training_dynamics_vs_infeasibility": training_dynamics_relation(),
    }
    return emit(
        "fallback_gradients",
        payload,
        config=base_record["config"],
        seeds=base_record["seeds"],
        script=SCRIPT,
    )


if __name__ == "__main__":
    path = main()
    print(path)
