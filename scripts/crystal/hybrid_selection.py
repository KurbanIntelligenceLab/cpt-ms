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

SCRIPT = "scripts/crystal/hybrid_selection.py"
FRONTIER_TARGETS = [1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0, 13.0, 16.0, 20.0]
N_VALUES = [1, 2, 4, 8, 16]


def main(ckpt: str, targets: list[float] | None = None, pool: int = 512,
         seed: int = 0, smoke: bool = False) -> Path:
    from scripts.train.train_crystal import (CrystalTrainConfig, CrystalVocab, ELEMENTS,
                                             GrammarMask, sample)

    tg = targets or (FRONTIER_TARGETS[:3] if smoke else FRONTIER_TARGETS)
    n_vals = [1, 2] if smoke else N_VALUES
    pool = 64 if smoke else pool

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, dev)
    cfg = CrystalTrainConfig(**ck["train_config"])
    V = CrystalVocab(ck.get("elements", ELEMENTS), order=ck["vocab_order"])
    gm = GrammarMask(V, order=ck["vocab_order"])
    max_len = int(ck["max_len"])

    rng2 = np.random.default_rng(seed + 1)
    rng3 = np.random.default_rng(seed + 2)
    trials = 40 if smoke else 200

    rows, diversity = [], []
    wall_pools = {}
    n_valid_pools = {}
    for t in tg:
        cfg.density_target = float(t)
        rng = np.random.default_rng(seed)
        t0 = time.time()
        seqs, _diag = sample(model, V, gm, cfg, dev, pool, use_mask=True,
                             project_at_inference=True, max_len=max_len + 2, rng=rng)
        wall_pools[float(t)] = time.time() - t0

        dens_list, formulas = [], []
        for sq in seqs:
            d = V.decode_density(sq)
            if d is None or not np.isfinite(d) or not (0 < d < 100):
                continue
            toks = [V.tokens[i] for i in sq]
            els = [tk[3:] for tk in toks if tk.startswith("el:")]
            cns = [int(tk[4:]) for tk in toks if tk.startswith("cnt:")]
            f: dict[str, int] = {}
            for e, kk in zip(els, cns):
                f[e] = f.get(e, 0) + kk
            dens_list.append(d)
            formulas.append((d, tuple(sorted(f.items()))))
        dens = np.array(dens_list, dtype=float)
        n_valid_pools[float(t)] = int(dens.size)
        if dens.size == 0:
            raise RuntimeError(f"no valid densities in the steered pool at target {t}")

        for n in n_vals:
            errs = [float(np.min(np.abs(dens[rng2.choice(len(dens),
                                                         size=min(n, len(dens)),
                                                         replace=False)] - t)))
                    for _ in range(trials)]
            rows.append({
                "target": float(t),
                "n": int(n),
                "mean_best_error": float(np.mean(errs)),
                "sd_best_error": float(np.std(errs, ddof=1)) if trials > 1 else 0.0,
                "trials": trials,
                "inference_samples_per_output": int(n),
            })

            picks = []
            for _ in range(trials):
                idx = rng3.choice(len(formulas), size=min(n, len(formulas)), replace=False)
                picks.append(min((formulas[i] for i in idx),
                                 key=lambda r: abs(r[0] - t))[1])
            diversity.append({
                "target": float(t), "n": int(n),
                "unique_formula_fraction": len(set(picks)) / len(picks),
                "n_outputs": len(picks),
            })

    payload = {
        "arm": "projection + best-of-n selection: n steered samples per target, keep closest on density",
        "checkpoint": Path(ckpt).name,
        "trained_density_target": float(ck["trained_density_target"]),
        "pool_size": int(pool),
        "n_valid_in_pool": {str(k): v for k, v in n_valid_pools.items()},
        "pool_wall_clock_s": {str(k): round(v, 1) for k, v in wall_pools.items()},
        "n_values": n_vals,
        "rows": rows,
        "protocol_note": (
            "One steered pool (project_at_inference=True at the requested target) is drawn "
            "once per target and reused for every n; selection is estimated by repeated "
            "subsampling of n draws from that pool, mirroring scripts/crystal/best_of_n.py's "
            "protocol exactly so the two arms are comparable. Unlike best_of_n.py, a fresh "
            "pool is drawn PER TARGET here because the pool itself is steered toward that "
            "target rather than drawn once from an unsteered model."),
        "diversity": diversity,
        "diversity_note": (
            "Unique-formula fraction among the selected outputs, computed identically to "
            "scripts/crystal/best_of_n.py's diversity block (same selection-then-count-uniques "
            "procedure) so the hybrid, best-of-n, and projection-alone (n=1 row here) arms sit "
            "on the same diversity definition."),
    }

    return emit("hybrid_selection" + ("_smoke" if smoke else ""), payload,
                config={"pool": pool, "targets": tg, "n_values": n_vals,
                        "checkpoint": Path(ckpt).name},
                seeds=[seed], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to a trained checkpoint; produce one with "
                         "python -m scripts.train.train_crystal --arm projected --ckpt <path>")
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(ckpt=a.ckpt, pool=a.pool, seed=a.seed, smoke=a.smoke))
