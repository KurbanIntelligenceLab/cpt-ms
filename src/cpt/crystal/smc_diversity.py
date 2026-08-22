from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cpt.common.artifacts import emit
from cpt.crystal.retarget import load_checkpoint
from cpt.crystal.smc_steering import smc

SCRIPT = "src/cpt/crystal/smc_diversity.py"


def main(ckpt: str, K: int, sigma: float, targets: list[float], trials: int = 100,
         seed: int = 0, smoke: bool = False) -> Path:
    from cpt.crystal.model import CrystalTrainConfig, GrammarMask
    from cpt.crystal.mp20 import ELEMENTS, CrystalVocab

    tg = targets[:2] if smoke else targets
    trials = 4 if smoke else trials

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, dev)
    cfg = CrystalTrainConfig(**ck["train_config"])
    V = CrystalVocab(ck.get("elements", ELEMENTS), order=ck["vocab_order"])
    gm = GrammarMask(V, order=ck["vocab_order"])
    max_len = int(ck["max_len"]) + 2

    rows, t0 = [], time.time()
    for t in tg:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        errs, formulas = [], []
        for _ in range(trials):
            seq, _d = smc(model, V, gm, cfg, dev, K, float(t), max_len, rng, sigma=float(sigma))
            d = V.decode_density(seq)
            if d is None or not np.isfinite(d) or not (0 < d < 100):
                continue
            errs.append(abs(d - float(t)))
            toks = [V.tokens[i] for i in seq]
            els = [tk[3:] for tk in toks if tk.startswith("el:")]
            cns = [int(tk[4:]) for tk in toks if tk.startswith("cnt:")]
            f: dict[str, int] = {}
            for e, kk in zip(els, cns, strict=False):
                f[e] = f.get(e, 0) + kk
            formulas.append(tuple(sorted(f.items())))
        rows.append({
            "target": float(t), "K": int(K), "sigma": float(sigma),
            "trials": trials, "n_valid": len(errs),
            "mean_error": float(np.mean(errs)) if errs else None,
            "sd_error": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
            "unique_formula_fraction": (len(set(formulas)) / len(formulas)) if formulas else None,
            "n_outputs": len(formulas),
        })

    payload = {
        "arm": "SMC steering, tuned sigma, diversity measured over independent runs",
        "checkpoint": Path(ckpt).name,
        "K": int(K),
        "sigma": float(sigma),
        "trials": trials,
        "rows": rows,
        "mean_error_over_targets": float(np.mean([r["mean_error"] for r in rows if r["mean_error"] is not None])),
        "mean_unique_formula_fraction_over_targets": float(np.mean(
            [r["unique_formula_fraction"] for r in rows if r["unique_formula_fraction"] is not None])),
        "protocol_note": (
            "Each row is `trials` INDEPENDENT SMC runs (K forward passes each), one accepted "
            "output per run. Unique-formula-fraction is measured across those independent "
            "accepted outputs. This is the closest like-for-like counterpart to best_of_n.py's "
            "diversity definition available for SMC: SMC has no larger pool to subsample from "
            "(each run already spends its full K-particle budget to produce one output), so "
            "diversity here is measured across repeated independent runs at the tuned budget "
            "rather than by resampling within one pool."),
        "wall_clock_s": round(time.time() - t0, 1),
    }

    return emit("smc_diversity" + ("_smoke" if smoke else ""), payload,
                config={"K": K, "sigma": sigma, "targets": tg, "trials": trials,
                        "checkpoint": Path(ckpt).name},
                seeds=[seed], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to a trained checkpoint; produce one with "
                         "python -m cpt.train.train_crystal --arm projected --ckpt <path>")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--targets", default="1.5,2.0,3.0,5.0,6.317,8.0,10.0")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tg = [float(x) for x in a.targets.split(",")]
    print(main(ckpt=a.ckpt, K=a.K, sigma=a.sigma, targets=tg, trials=a.trials,
              seed=a.seed, smoke=a.smoke))
