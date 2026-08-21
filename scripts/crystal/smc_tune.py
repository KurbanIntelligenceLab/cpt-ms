from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.retarget import load_checkpoint
from scripts.crystal.smc_steering import smc, FRONTIER_TARGETS

SCRIPT = "scripts/crystal/smc_tune.py"
SIGMAS_WIDE = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
K_FIXED = 16


def main(ckpt: str, targets=None, K: int = K_FIXED, sigmas=None, runs: int = 16,
         seed: int = 0, smoke: bool = False) -> Path:
    from scripts.crystal.mp20 import ELEMENTS, CrystalVocab
    from scripts.train.train_crystal import CrystalTrainConfig, GrammarMask

    tg = targets or (FRONTIER_TARGETS[:3] if smoke else FRONTIER_TARGETS)
    sigs = sigmas or ([0.3, 1.0, 2.0] if smoke else SIGMAS_WIDE)
    runs = 4 if smoke else runs

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, dev)
    cfg = CrystalTrainConfig(**ck["train_config"])
    V = CrystalVocab(ck.get("elements", ELEMENTS), order=ck["vocab_order"])
    gm = GrammarMask(V, order=ck["vocab_order"])
    max_len = int(ck["max_len"]) + 2

    rows, t0 = [], time.time()
    for t in tg:
        for sig in sigs:
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            errs, valid = [], 0
            tw0 = time.time()
            for _ in range(runs):
                seq, _d = smc(model, V, gm, cfg, dev, K, float(t), max_len, rng, sigma=float(sig))
                d = V.decode_density(seq)
                if d is not None and np.isfinite(d) and 0 < d < 100:
                    valid += 1
                    errs.append(abs(d - float(t)))
            wall = time.time() - tw0
            rows.append({
                "target": float(t), "K": int(K), "sigma": float(sig), "runs": runs,
                "n_valid": valid,
                "mean_error": float(np.mean(errs)) if errs else None,
                "sd_error": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
                "wall_clock_s_per_output": wall / runs,
                "forward_passes_per_output": int(K),
            })

    best_per_target = {}
    for r in rows:
        cur = best_per_target.get(r["target"])
        if r["mean_error"] is not None and (cur is None or r["mean_error"] < cur["mean_error"]):
            best_per_target[r["target"]] = r
    best_rows = [dict(v) for _, v in sorted(best_per_target.items())]

    per_sigma_mean = {}
    for sig in sigs:
        errs = [r["mean_error"] for r in rows if r["sigma"] == sig and r["mean_error"] is not None]
        per_sigma_mean[str(sig)] = float(np.mean(errs)) if errs else None
    global_best_sigma = min((s for s in sigs if per_sigma_mean[str(s)] is not None),
                            key=lambda s: per_sigma_mean[str(s)])

    old = None
    old_path = Path("results/smc_steering.json")
    if old_path.exists():
        import json
        old = json.loads(old_path.read_text())["payload"].get("sigma_sweep")

    payload = {
        "arm": "sequential Monte Carlo steering at inference, potential width tuned",
        "checkpoint": Path(ckpt).name,
        "trained_density_target": float(ck["trained_density_target"]),
        "K": int(K),
        "sigmas": sigs,
        "runs_per_cell": runs,
        "rows": rows,
        "per_sigma_mean_error_over_targets": per_sigma_mean,
        "global_best_fixed_sigma": {"sigma": float(global_best_sigma),
                                    "mean_error_over_targets": per_sigma_mean[str(global_best_sigma)]},
        "best_per_target_selected_on_eval": best_rows,
        "tuning_selection_note": (
            "best_per_target_selected_on_eval selects sigma per target from the same runs "
            "that report the error, so its mean is optimistically biased and is not the "
            "reported comparison point; global_best_fixed_sigma is the single sigma value "
            "chosen without looking at per-target outcomes and is the honest tuned number, "
            "matching the convention results/smc_steering.json set for its own sweep."),
        "matched_particle_budget_note": (
            f"K is held fixed at {K} across every sigma in this sweep, identical to the "
            "existing sigma_sweep in results/smc_steering.json, so differences across "
            "rows are attributable to the potential width alone, not to particle count."),
        "prior_sweep_reference": {
            "existing_sigma_sweep_sigmas": (old.get("sigmas") if old else None),
            "existing_sigma_sweep_finding": (old.get("finding") if old else None),
        },
        "still_untuned": (
            "proposal distribution, resampling threshold, and number of resampling points "
            "remain untuned; this sweep only tunes the potential width sigma at fixed K, "
            "over a range roughly 30x wider than the two-point probe already on record"),
    }

    return emit("smc_tuned" + ("_smoke" if smoke else ""), payload,
                config={"K": K, "sigmas": sigs, "targets": tg, "runs": runs,
                        "checkpoint": Path(ckpt).name},
                seeds=[seed], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="development/ckpt/projected_d5.0.pt")
    ap.add_argument("--K", type=int, default=K_FIXED)
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(ckpt=a.ckpt, K=a.K, runs=a.runs, seed=a.seed, smoke=a.smoke))
