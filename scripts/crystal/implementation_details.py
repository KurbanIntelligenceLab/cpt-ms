from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

from scripts.common.arms import resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.layer import projection as projmod

SCRIPT = "scripts/crystal/implementation_details.py"
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())["payload"]


def _crystal_arm(arm: str) -> dict:
    seeds = [0, 1, 2]
    runs = [_load(f"crystal_{arm}_seed{s}") for s in seeds]
    base_cfg = {k: v for k, v in runs[0]["train_config"].items() if k != "seed"}
    for r in runs:
        cfg = {k: v for k, v in r["train_config"].items() if k != "seed"}
        if cfg != base_cfg:
            raise ValueError(f"arm {arm}: train_config differs across seeds")
    return {
        "train_config_shared": base_cfg,
        "seeds": [r["train_config"]["seed"] for r in runs],
        "device": runs[0]["device"],
        "gpu": runs[0]["gpu"],
        "n_parameters": runs[0]["n_parameters"],
        "vocab_size": runs[0]["vocab_size"],
        "n_train_structures": runs[0]["n_train_structures"],
        "max_len": runs[0]["max_len"],
        "wall_clock_s_per_seed": {str(r["train_config"]["seed"]): r["wall_clock_s"] for r in runs},
        "wall_clock_s_mean": float(np.mean([r["wall_clock_s"] for r in runs])),
        "source_artifacts": [f"results/crystal_{arm}_seed{s}.json" for s in seeds],
    }


def _synthetic_arm(arm: str) -> dict:
    seeds = [0, 1, 2]
    runs = [_load(f"train_{arm}_seed{s}") for s in seeds]
    base_cfg = {k: v for k, v in runs[0]["train_config"].items() if k != "seed"}
    for r in runs:
        cfg = {k: v for k, v in r["train_config"].items() if k != "seed"}
        if cfg != base_cfg:
            raise ValueError(f"synthetic arm {arm}: train_config differs across seeds")
    return {
        "train_config_shared": base_cfg,
        "task_config": runs[0]["task_config"],
        "device": runs[0]["device"],
        "gpu": runs[0]["gpu"],
        "n_parameters": runs[0]["n_parameters"],
        "seeds": [r["train_config"]["seed"] for r in runs],
        "wall_clock_s_per_seed": {str(r["train_config"]["seed"]): r["wall_clock_s"] for r in runs},
        "wall_clock_s_mean": float(np.mean([r["wall_clock_s"] for r in runs])),
        "source_artifacts": [f"results/train_{arm}_seed{s}.json" for s in seeds],
    }


def _solver_settings() -> dict:
    sig = inspect.signature(projmod.project)
    defaults = {
        "max_iter": sig.parameters["max_iter"].default,
        "tol": sig.parameters["tol"].default,
        "damping": sig.parameters["damping"].default,
        "fast_hessian_dtype": str(sig.parameters["fast_hessian_dtype"].default),
        "solve_dtype": "torch.float64 (hard-coded, not a parameter)",
        "backtracking_iterations": 30,
        "backtracking_acceptance_slack": "1e-12 * |g0|, floor 1e-14",
        "convergence_check_dtype": "torch.float64, before output cast; tol floor max(tol, 1e-8) on the residual check",
    }
    tolerances = _load("gradient_verification")["tolerances"]
    return {
        "defaults_from": "scripts/layer/projection.py:project()",
        "defaults": defaults,
        "achieved_residuals": {
            "worst_forward_residual": tolerances["worst_forward_residual"],
            "worst_rel_diff_vs_finite_difference": tolerances["worst_rel_diff_vs_finite_difference"],
            "worst_rel_diff_vs_cvxpylayers": tolerances["worst_rel_diff_vs_cvxpylayers"],
            "cvxpylayers_configs_compared": tolerances["cvxpylayers_configs_compared"],
        },
    }


def _tokenizer() -> dict:
    return {
        "crystal": {
            "level": "line-level pseudo-token stream (not per-character); the grammar mask itself operates per-character over a CIF skeleton (scripts/crystal/cif_grammar.py) but the trained model's vocabulary is the coarser CrystalVocab token set",
            "vocab_size_default": 500,
            "vocab_size_conditional_smoke_variant": 884,
            "vocab_composition": {
                "special": ["<BOS>", "<COMP>", "<END_COMP>", "<CELL>", "<END_CELL>", "<EOS>"],
                "elements": "one token per element in ELEMENTS (94 elements, H through Pu)",
                "counts": "cnt:1 .. cnt:N_CNT (N_CNT=16)",
                "cell_lengths": "len:0 .. len:N_LEN-1, uniform bins over [LEN_LO, LEN_HI]",
                "cell_angles": "ang:0 .. ang:N_ANG-1, uniform bins over [ANG_LO, ANG_HI]",
            },
            "main_text_runs_resolution": {"N_LEN": 256, "N_ANG": 128, "N_CNT": 16, "resulting_vocab": 500},
            "current_source_default_resolution": {"N_LEN": 512, "N_ANG": 256, "N_CNT": 16,
                                                    "resulting_vocab_at_current_ranges": 884},
            "bin_ranges_current": {"LEN_LO": 2.0, "LEN_HI": 46.0, "ANG_LO": 5.0, "ANG_HI": 175.0},
            "resolution_rationale": (
                "chosen from measured MP-20 median relative discretization error, not convention: "
                "64/32 bins -> 4.4%, 128/64 -> 2.4%, 256/128 -> 1.1%, 512/256 -> 0.6% (all measured under "
                "an earlier, narrower bin range). All crystal_A{1,2,3}_seed*.json runs used the "
                "256/128 setting and its earlier range, giving vocab_size=500 as recorded in every one "
                "of those artifacts. A later fix widened the bin range to cover the full MP-20 support "
                "after a clamping defect was found (bin_index_caveat, below) and re-set resolution to "
                "512/256 together with the new range, giving an 884-token vocabulary with 0.84% overall "
                "error; crystal_A4_smoke.json is the one artifact in this repository built with that "
                "corrected tokenizer."
            ),
            "serialization_order": "cell_first (default) or comp_first; cell_first is required for the density constraint to be expressible as a per-step linear budget, since density is mass/volume and the volume must close before the mass channel can be evaluated",
            "bin_index_caveat": (
                "len:/ang: token payloads are BIN INDICES, not physical values; a past defect in "
                "scripts/train/train_crystal.py read them as floats directly, inflating the decoded cell "
                "volume by a factor of hundreds and silently producing a plausible-looking but wrong "
                "budget. The budget computation now always maps the index through the bin centre "
                "(V.LEN_LO + (j + 0.5) * (V.LEN_HI - V.LEN_LO) / V.N_LEN) before using it."
            ),
        },
        "synthetic": {
            "level": "single-character-equivalent, one token per symbol",
            "vocab_size": 4,
            "symbols": ["A", "B", "C", "D"],
            "charges": [2, 1, -1, -2],
            "sequence_length": 8,
            "vocab_source": "scripts/synthetic/task.py TaskConfig.symbols/charges",
        },
    }


def main() -> Path:
    crystal_arms = {arm: _crystal_arm(arm) for arm in ("unconstrained", "mask_only", "projected")}
    crystal_a3_f2 = _load("crystal_projected_cell_first_seed0")
    crystal_a4_smoke = _load("crystal_conditional_smoke")

    synthetic_arms = {arm: _synthetic_arm(arm) for arm in ("unconstrained", "mask_only", "projected")}
    synthetic_summary = _load("training_summary")

    cb = _load("conditional_baseline")
    a3_walls = [v["wall"] for v in cb["runs"].values() if resolve(v["arm"]) == "projected"]
    a4_walls = [v["wall"] for v in cb["runs"].values() if resolve(v["arm"]) == "conditional"]

    overhead = _load("overhead_benchmark")

    ratio_crystal_a3_over_a2_single_ckpt = (
        crystal_arms["projected"]["wall_clock_s_mean"] / crystal_arms["mask_only"]["wall_clock_s_mean"]
    )
    ratio_crystal_a3_over_a1_single_ckpt = (
        crystal_arms["projected"]["wall_clock_s_mean"] / crystal_arms["unconstrained"]["wall_clock_s_mean"]
    )
    ratio_synthetic_a3_over_a2 = (
        synthetic_arms["projected"]["wall_clock_s_mean"] / synthetic_arms["mask_only"]["wall_clock_s_mean"]
    )

    payload = {
        "crystal_arms": crystal_arms,
        "crystal_variant_runs": {
            "projected_cell_first_static_channel": {
                "train_config": crystal_a3_f2["train_config"],
                "device": crystal_a3_f2["device"], "gpu": crystal_a3_f2["gpu"],
                "n_parameters": crystal_a3_f2["n_parameters"], "vocab_size": crystal_a3_f2["vocab_size"],
                "wall_clock_s": crystal_a3_f2["wall_clock_s"],
                "note": "same architecture and vocab as the main projected arm; wall clock differs because this run includes the joint two-constraint solve path",
                "source_artifact": "results/crystal_projected_cell_first_seed0.json",
            },
            "conditional_smoke": {
                "train_config": crystal_a4_smoke["train_config"],
                "device": crystal_a4_smoke["device"], "gpu": crystal_a4_smoke["gpu"],
                "n_parameters": crystal_a4_smoke["n_parameters"], "vocab_size": crystal_a4_smoke["vocab_size"],
                "wall_clock_s": crystal_a4_smoke["wall_clock_s"],
                "note": "884-token vocab, CPU device, 300 steps, filter_tol 0.5 -- a smoke-test configuration, not a reported result",
                "source_artifact": "results/crystal_conditional_smoke.json",
            },
        },
        "synthetic_arms": synthetic_arms,
        "synthetic_training_summary": {
            "steps": synthetic_summary["steps"],
            "wall_clock_s_mean": synthetic_summary["wall_clock_s_mean"],
            "wall_clock_ratio_A3_over_A2_published": synthetic_summary["wall_clock_ratio_A3_over_A2"],
            "wall_clock_ratio_A3_over_A2_recomputed_from_per_seed": ratio_synthetic_a3_over_a2,
            "source_artifact": "results/training_summary.json",
        },
        "tokenizer": _tokenizer(),
        "solver": _solver_settings(),
        "conditional_baseline": {
            "note": cb["note"],
            "per_target": cb["per_target"],
            "summary": cb["summary"],
            "wall_clock_s": {
                "projected_per_run": a3_walls,
                "projected_mean": float(np.mean(a3_walls)),
                "projected_total_8_runs": float(np.sum(a3_walls)),
                "conditional_per_run": a4_walls,
                "conditional_mean": float(np.mean(a4_walls)),
                "conditional_total_8_runs": float(np.sum(a4_walls)),
                "hardware_note": (
                    "the conditional_baseline.json record itself was written on a "
                    "macOS/arm64 host (record platform field), which is inconsistent with the "
                    "per-run wall clocks; this artifact carries the wall-clock figures exactly as "
                    "stored and does not alter them, but the hardware for THESE 16 runs is not "
                    "independently confirmed to be the RTX 5090 recorded on the crystal_A*_seed*.json "
                    "single-checkpoint runs, unlike those, which explicitly stamp gpu='NVIDIA GeForce RTX 5090'"
                ),
            },
            "source_artifact": "results/conditional_baseline.json",
        },
        "overhead_benchmark": {
            "superseded_note": overhead["superseded"]["note"],
            "hardware": overhead["hardware"],
            "summary": overhead["summary"],
            "training_step_context": overhead["training_step_context"],
            "actionable_scoping": overhead["actionable_scoping"],
            "fp64_penalty": overhead["fp64_penalty"],
            "verdict": overhead["verdict"],
            "source_artifact": "results/overhead_benchmark.json",
        },
        "compute_normalized_comparison": {
            "what_is_matched": (
                "The crystal unconstrained/mask-only/projected single-checkpoint runs (results/crystal_{unconstrained,mask-only,projected}_seed{0,1,2}.json) "
                "share architecture (d_model=256, 6 layers, 8 heads, 5,002,228 parameters), vocabulary "
                "(500 tokens), step count (6000), batch size (128), optimizer, schedule, training data "
                "(26,910 structures) and hardware (cuda, NVIDIA GeForce RTX 5090); they differ only in "
                "whether the mask (mask-only) or mask-plus-trained-through-projection (projected) is applied, so the "
                "projected/mask-only wall-clock ratio computed from these runs isolates the projection layer's marginal "
                "training cost on identical hardware and identical step count."
            ),
            "what_is_not_matched": (
                "The results/conditional_baseline.json sweep (projected vs conditional, 8 runs each across 4 targets x "
                "2 seeds) is NOT matched to the single-checkpoint runs on hardware: its stored JSON record "
                "carries platform='macOS-26.5.2-arm64-arm-64bit', while every crystal_A*_seed*.json run "
                "explicitly stamps device='cuda', gpu='NVIDIA GeForce RTX 5090'. Step count is also not "
                "confirmed identical between projected and conditional inside that sweep from the stored payload (train_config "
                "per run is not recorded in the runs dict, only loss/wall/target/seed/n_train), so the projected/conditional "
                "wall-clock ratio from conditional_baseline.json is NOT presented as a controlled overhead "
                "measurement -- it is reported as the raw cost of the two training regimes as actually run, "
                "with the hardware caveat stated explicitly wherever it is used."
            ),
            "single_checkpoint_vs_mask_only": {
                "crystal_projected_over_mask_only_wall_clock_ratio": ratio_crystal_a3_over_a2_single_ckpt,
                "crystal_projected_over_unconstrained_wall_clock_ratio": ratio_crystal_a3_over_a1_single_ckpt,
                "synthetic_projected_over_mask_only_wall_clock_ratio_published": synthetic_summary["wall_clock_ratio_A3_over_A2"],
                "note": "crystal ratio is smaller than the synthetic 6.0x figure reported in si:overhead because the crystal backbone (5.0M params, 6000 steps) is far larger than the per-step projection overhead relative to its own forward/backward cost, whereas the synthetic backbone (0.8M params, 4000 steps) is small enough that the projection solve dominates a larger fraction of each step",
            },
            "cost_structure_headline": {
                "claim": "one projection-trained checkpoint (projected), deployed at inference time across a family of targets with no retraining, versus one conditionally-trained checkpoint (conditional) PER target",
                "single_checkpoint_training_wall_clock_s": crystal_arms["projected"]["wall_clock_s_mean"],
                "n_conditional_checkpoints_for_a_7_target_family": 7,
                "conditional_baseline_mean_wall_clock_s_per_checkpoint": float(np.mean(a4_walls)),
                "conditional_baseline_total_wall_clock_s_for_7_checkpoints_at_measured_mean": 7 * float(np.mean(a4_walls)),
                "ratio_conditional_7x_over_single_projection_checkpoint": (
                    7 * float(np.mean(a4_walls)) / crystal_arms["projected"]["wall_clock_s_mean"]
                ),
                "retargeting_inference_overhead_s": {
                    "three_checkpoints_x_ten_targets_total_s": 241,
                    "one_training_run_s_approx": 1660,
                    "source": "reports/retargeting_notes.md; results/zero_shot_retargeting.json (per-checkpoint wall_clock_s)",
                },
                "caveat": (
                    "the conditional mean wall clock above (216.45 s) is read from conditional_baseline.json, whose "
                    "record is stamped with a different platform than the RTX-5090 single-checkpoint runs (see "
                    "what_is_not_matched); it is used here only because it is the sole measured conditional training cost "
                    "in the repository, and the ratio is reported as an as-measured cost-structure figure, not a "
                    "hardware-controlled benchmark"
                ),
            },
        },
    }

    return emit(
        "implementation_details",
        payload,
        config={"reviewer_points": ["W4", "W9", "Q2"]},
        seeds=[0, 1, 2],
        script=SCRIPT,
    )


if __name__ == "__main__":
    print(main())
