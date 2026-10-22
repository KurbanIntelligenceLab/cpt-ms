from __future__ import annotations

import numpy as np
import torch

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
