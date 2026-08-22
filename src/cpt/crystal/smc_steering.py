"""Sequential Monte Carlo steering: the principled inference-time alternative."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cpt.common.artifacts import emit
from cpt.crystal.retarget import load_checkpoint

SCRIPT = "src/cpt/crystal/smc_steering.py"
FRONTIER_TARGETS = [1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0, 13.0, 16.0, 20.0]
K_VALUES = [1, 4, 16, 64]
AMU_PER_A3_TO_G_CM3 = 1.6605390666


def _density_estimate(prefix_ids: np.ndarray, V, cfg) -> np.ndarray:
    from cpt.crystal.mp20 import ATOMIC_MASS

    out = np.full(prefix_ids.shape[0], np.nan)
    for i in range(prefix_ids.shape[0]):
        toks = [V.tokens[t] for t in prefix_ids[i]]
        formula, lens, angs, cur = {}, [], [], None
        for t in toks:
            if t.startswith("el:"):
                cur = t[3:]
            elif t.startswith("cnt:") and cur:
                formula[cur] = formula.get(cur, 0) + int(t[4:])
                cur = None
            elif t.startswith("len:"):
                j = int(t[4:])
                lens.append(V.LEN_LO + (j + 0.5) * (V.LEN_HI - V.LEN_LO) / V.N_LEN)
            elif t.startswith("ang:"):
                j = int(t[4:])
                angs.append(V.ANG_LO + (j + 0.5) * (V.ANG_HI - V.ANG_LO) / V.N_ANG)
        if len(lens) != 3 or len(angs) != 3 or not formula:
            continue
        vol = V.cell_volume(lens, angs)
        if not vol or vol <= 0:
            continue
        mass = sum(ATOMIC_MASS.get(e, 0.0) * n for e, n in formula.items())
        out[i] = mass / vol * AMU_PER_A3_TO_G_CM3
    return out


def smc(model, V, gm, cfg, device, K: int, target: float, max_len: int,
        rng, sigma: float = 1.0) -> tuple[list[int], dict]:
    """One SMC run with K particles. Returns the surviving sequence and diagnostics."""

    eos = V.index["<EOS>"]
    ids = torch.full((K, 1), V.index["<BOS>"], dtype=torch.long, device=device)
    done = torch.zeros(K, dtype=torch.bool, device=device)
    logw = torch.zeros(K, device=device)
    n_resample = 0

    for _ in range(max_len - 1):
        with torch.no_grad():
            logits = model(ids)[:, -1, :]
        cur = ids.cpu().numpy()
        padded = np.concatenate([cur, np.zeros((cur.shape[0], 1), dtype=cur.dtype)], axis=1)
        mk = torch.from_numpy(gm.masks_for(padded)[:, -1, :]).to(device)
        mk[done] = False
        mk[done, eos] = True
        logits = logits.masked_fill(~mk, float("-inf"))
        probs = torch.softmax(logits, -1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        done = done | (nxt.squeeze(-1) == eos)

        tok_names = [V.tokens[int(t)] for t in nxt.squeeze(-1).cpu().numpy()]
        if any(t.startswith("cnt:") for t in tok_names) and not bool(done.all()):
            est = _density_estimate(ids.cpu().numpy(), V, cfg)
            good = np.isfinite(est)
            if good.sum() >= 2:
                pot = np.where(good, -((est - target) ** 2) / (2 * sigma ** 2), -1e3)
                logw = logw + torch.tensor(pot, device=device, dtype=logw.dtype)
                w = torch.softmax(logw, 0)
                ess = 1.0 / float((w ** 2).sum())
                if ess < K / 2:
                    idx = torch.multinomial(w, K, replacement=True)
                    ids = ids[idx]
                    done = done[idx]
                    logw = torch.zeros(K, device=device)
                    n_resample += 1
        if bool(done.all()):
            break

    w = torch.softmax(logw, 0)
    pick = int(torch.multinomial(w, 1).item())
    return [int(t) for t in ids[pick].cpu().numpy()], {"n_resample": n_resample}


SIGMAS = [0.3, 1.0]


def main(ckpt: str, targets=None, k_values=None, runs: int = 32,
         seed: int = 0, smoke: bool = False, sigmas=None) -> Path:
    from cpt.crystal.model import CrystalTrainConfig, GrammarMask
    from cpt.crystal.mp20 import ELEMENTS, CrystalVocab

    tg = targets or (FRONTIER_TARGETS[:2] if smoke else FRONTIER_TARGETS)
    ks = k_values or ([1, 4] if smoke else K_VALUES)
    runs = 4 if smoke else runs

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, dev)
    cfg = CrystalTrainConfig(**ck["train_config"])
    V = CrystalVocab(ck.get("elements", ELEMENTS), order=ck["vocab_order"])
    gm = GrammarMask(V, order=ck["vocab_order"])
    max_len = int(ck["max_len"]) + 2

    sigs = sigmas or ([1.0] if smoke else SIGMAS)
    rows = []
    for t in tg:
        for K in ks:
            for sig in sigs:
                rng = np.random.default_rng(seed)
                torch.manual_seed(seed)
                errs, valid, t0 = [], 0, time.time()
                for _ in range(runs):
                    seq, _d = smc(model, V, gm, cfg, dev, K, float(t), max_len, rng,
                                  sigma=float(sig))
                    d = V.decode_density(seq)
                    if d is not None and np.isfinite(d) and 0 < d < 100:
                        valid += 1
                        errs.append(abs(d - float(t)))
                wall = time.time() - t0
                rows.append({
                    "target": float(t), "K": int(K), "sigma": float(sig), "runs": runs,
                    "n_valid": valid,
                    "mean_error": float(np.mean(errs)) if errs else None,
                    "sd_error": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
                    "wall_clock_s_per_output": wall / runs,
                    "forward_passes_per_output": int(K),
                })

    best = {}
    for r in rows:
        key = (r["target"], r["K"])
        cur = best.get(key)
        if r["mean_error"] is not None and (cur is None or r["mean_error"] < cur["mean_error"]):
            best[key] = r
    best_rows = [{"target": k[0], "K": k[1], "sigma": v["sigma"],
                  "mean_error": v["mean_error"], "sd_error": v["sd_error"]}
                 for k, v in sorted(best.items())]

    payload = {
        "arm": "sequential Monte Carlo steering at inference, untuned",
        "checkpoint": Path(ckpt).name,
        "trained_density_target": float(ck["trained_density_target"]),
        "k_values": ks,
        "sigmas": sigs,
        "rows": rows,
        "best_by_sigma": best_rows,
        "sigma_note": ("the potential's width is the dominant tuning knob; reporting only a "
                       "fixed sigma would have made an implementation detail look like a "
                       "property of SMC"),
        "potential": ("Gaussian on the prefix's expected final density against the target, "
                      "sigma = 1.0 g/cm^3, evaluated at element positions"),
        "resampling": "multinomial, triggered when effective sample size falls below K/2",
        "estimate_note": ("the density estimate uses the trainer's own budget function, so "
                          "the quantity SMC steers cannot disagree with the quantity the "
                          "projection arm targets"),
        "lower_bound_note": ("no proposal is learned and no threshold is tuned, so these "
                             "numbers are a lower bound on what SMC can achieve rather than "
                             "its ceiling; the purpose is to locate the crossover for a "
                             "competent untuned implementation"),
        "cost_note": ("K forward passes per accepted output at every target, indefinitely, "
                      "against one training run and one forward pass for the projection"),
    }
    return emit("smc_steering" + ("_smoke" if smoke else ""), payload,
                config={"k_values": ks, "targets": tg, "runs": runs,
                        "checkpoint": Path(ckpt).name},
                seeds=[seed], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to a trained checkpoint; produce one with "
                         "python -m cpt.train.train_crystal --arm projected --ckpt <path>")
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(ckpt=a.ckpt, runs=a.runs, seed=a.seed, smoke=a.smoke))
