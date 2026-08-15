"""Best-of-n from the unsteered model: the baseline a reviewer asks for first.

WHY THIS IS MANDATORY. The amortization claim has so far been argued against conditional
training, which needs a model per target. Best-of-n needs no training at all: sample n
completions from the masked-but-unsteered model and keep whichever lands closest to the
target. It amortizes perfectly across targets by construction -- one model, any target -- so
it is the cheapest thing that does what this paper claims to do, and the paper is weaker
until it is measured rather than dismissed.

WHAT IT COSTS AND WHY THAT IS THE POINT. Best-of-n pays at inference, every time, in
proportion to n. The projection pays once at training. So the two are not points on the same
axis and the comparison is a curve against a point: the question is what n the unsteered model
needs to match one steered sample, and whether that n is affordable at the targets that matter.
Reporting the accuracy alone would flatter whichever arm the axis was chosen to favour.

SELECTION IS FROM THE SAME POOL. Best-of-n draws from the A3 checkpoint's BACKBONE reading
(mask on, projection off), not from a separately trained model, so the comparison isolates
selection-at-inference against projection-at-training with the weights held fixed.
"""
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

SCRIPT = "scripts/crystal/best_of_n.py"
FRONTIER_TARGETS = [1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0, 13.0, 16.0, 20.0]
N_VALUES = [1, 4, 16, 64]


def main(ckpt: str, targets: list[float] | None = None, pool: int = 512,
         seed: int = 0, smoke: bool = False) -> Path:
    from scripts.train.train_crystal import (CrystalTrainConfig, CrystalVocab, ELEMENTS,
                                             GrammarMask, sample)

    tg = targets or (FRONTIER_TARGETS[:3] if smoke else FRONTIER_TARGETS)
    n_vals = [1, 4] if smoke else N_VALUES
    pool = 64 if smoke else pool

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, dev)
    cfg = CrystalTrainConfig(**ck["train_config"])
    V = CrystalVocab(ck.get("elements", ELEMENTS), order=ck["vocab_order"])
    gm = GrammarMask(V, order=ck["vocab_order"])
    max_len = int(ck["max_len"])

    # ONE pool of unsteered samples, reused for every target and every n. Drawing a fresh
    # pool per target would confound n with sampling noise, and best-of-n against a target
    # does not change how the samples were generated -- only which one is kept.
    t0 = time.time()
    rng = np.random.default_rng(seed)
    seqs, _ = sample(model, V, gm, cfg, dev, pool, use_mask=True,
                     project_at_inference=False, max_len=max_len + 2, rng=rng)
    wall_pool = time.time() - t0

    # Per-sample densities via the same public decoder structural_metrics uses. That
    # function returns aggregates only, and best-of-n needs the individual values; decoding
    # here rather than widening the shared function keeps every published aggregate on the
    # identical code path it was computed with.
    dens = np.array([d for d in (V.decode_density(s) for s in seqs)
                     if d is not None and np.isfinite(d) and 0 < d < 100], dtype=float)
    if dens.size == 0:
        raise RuntimeError("no valid densities in the sampled pool")

    rng2 = np.random.default_rng(seed + 1)
    rows = []
    trials = 40 if smoke else 200
    for t in tg:
        for n in n_vals:
            # Expected best-of-n error by resampling n draws from the pool: this estimates
            # what a practitioner running best-of-n would get, rather than reporting the
            # single best draw in the pool and calling it best-of-n.
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

    # THE COMPARISON THAT MATTERS IS NOT ACCURACY ALONE. Best-of-n buys accuracy by
    # DISCARDING samples, so the accepted set is the pool's tail near the target and its
    # diversity falls with n. Measuring only error against compute would hide that, and it
    # is the axis on which selection and shaping genuinely differ.
    formulas = []
    for sq in seqs:
        d = V.decode_density(sq)
        if d is None or not np.isfinite(d) or not (0 < d < 100):
            continue
        toks = [V.tokens[i] for i in sq]
        els = [t[3:] for t in toks if t.startswith("el:")]
        cns = [int(t[4:]) for t in toks if t.startswith("cnt:")]
        f: dict[str, int] = {}
        for e, kk in zip(els, cns):
            f[e] = f.get(e, 0) + kk
        formulas.append((d, tuple(sorted(f.items()))))

    rng3 = np.random.default_rng(seed + 2)
    diversity = []
    for t in tg:
        for n in n_vals:
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
        "arm": "best-of-n from the unsteered (masked, unprojected) model",
        "checkpoint": Path(ckpt).name,
        "trained_density_target": float(ck["trained_density_target"]),
        "pool_size": int(pool),
        "n_valid_in_pool": int(dens.size),
        "pool_wall_clock_s": round(wall_pool, 1),
        "pool_density": {"mean": float(dens.mean()), "sd": float(dens.std(ddof=1)),
                         "min": float(dens.min()), "max": float(dens.max())},
        "n_values": n_vals,
        "rows": rows,
        "protocol_note": (
            "One unsteered pool is drawn once and reused for every target and every n; "
            "best-of-n is estimated by repeated subsampling of n draws from that pool, so "
            "the reported value is the expected error of a practitioner running best-of-n "
            "rather than the best single draw in the pool."),
        "diversity": diversity,
        "diversity_note": (
            "Unique-formula fraction among best-of-n outputs. Selection concentrates the "
            "accepted set on the pool's tail near the target, so accuracy and diversity "
            "trade against each other in n. The projection shapes the distribution instead "
            "of filtering it, so its diversity does not depend on how hard the target is to "
            "hit -- which is the axis on which the two approaches actually differ."),
        "cost_note": (
            "Best-of-n costs n forward passes per accepted output, every time, at every "
            "target. The projection costs one training run and one forward pass per output. "
            "Accuracy alone is therefore not the comparison."),
    }

    # Best-of-n can only return what the pool contains: a target outside the pool span
    # is unreachable by selection at ANY n, which no accuracy-versus-compute curve shows.
    unreachable = [float(t) for t in tg if t > float(dens.max()) or t < float(dens.min())]
    payload["targets_outside_pool_support"] = unreachable
    payload["support_note"] = (
        f"The unsteered pool spans {dens.min():.2f} to {dens.max():.2f} g/cm^3. Best-of-n "
        f"cannot return a sample outside that range at any n, so the "
        f"{len(unreachable)} target(s) outside it are unreachable by selection rather than "
        f"merely expensive." if unreachable else
        f"Every target lies inside the pool's span of {dens.min():.2f} to {dens.max():.2f}.")

    return emit("WS6_best_of_n" + ("_smoke" if smoke else ""), payload,
                config={"pool": pool, "targets": tg, "n_values": n_vals,
                        "checkpoint": Path(ckpt).name},
                seeds=[seed], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/A3_d5.0.pt")
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(ckpt=a.ckpt, pool=a.pool, seed=a.seed, smoke=a.smoke))
