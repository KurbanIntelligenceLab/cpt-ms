from __future__ import annotations

import numpy as np
import torch

from cpt.crystal.channels import _element_position, _mass_budget, mass_channel
from cpt.crystal.mp20 import ATOMIC_MASS
from cpt.layer.projection import project


def pad_batch(seqs: list[list[int]], pad: int) -> np.ndarray:
    L = max(len(s) for s in seqs)
    a = np.full((len(seqs), L), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return a


@torch.no_grad()
def sample(model, V, gm, cfg, device, n, *, use_mask: bool, project_at_inference: bool,
           max_len: int, rng) -> tuple[list[list[int]], dict]:
    model.eval()
    gen_seed = int(rng.integers(0, 2 ** 63 - 1))
    torch.manual_seed(gen_seed)
    ids = torch.full((n, 1), V.index["<BOS>"], dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    n_infeasible = torch.zeros(n, device=device)
    n_projected = torch.zeros(n, device=device)
    n_feasible = torch.zeros(n, device=device)
    n_solver_fallback = torch.zeros(n, device=device)
    n_degenerate = torch.zeros(n, device=device)
    eos = V.index["<EOS>"]
    mass_t = torch.tensor([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                           for t in V.tokens], device=device, dtype=torch.float32)

    for _ in range(max_len - 1):
        logits = model(ids)[:, -1, :]
        if use_mask:
            cur = ids.cpu().numpy()
            padded = np.concatenate([cur, np.zeros((cur.shape[0], 1), dtype=cur.dtype)], axis=1)
            mk = torch.from_numpy(gm.masks_for(padded)[:, -1, :]).to(device)
            mk[done] = False
            mk[done, eos] = True
            logits = logits.masked_fill(~mk, float("-inf"))

        if project_at_inference:
            B, Vsz = logits.shape
            live = mk if use_mask else torch.ones_like(logits, dtype=torch.bool)
            budget, slots = _mass_budget(ids.cpu().numpy(), V, cfg)
            budget = torch.from_numpy(budget).to(device).float()
            slots = torch.from_numpy(slots).to(device).float()
            el_next = torch.from_numpy(_element_position(ids.cpu().numpy(), V)).to(device)
            active = (slots > 0) & torch.isfinite(budget) & ~done & el_next
            if cfg.channel == "prefix":
                ch = torch.from_numpy(mass_channel(ids.cpu().numpy(), V)).to(device)
            else:
                ch = mass_t.unsqueeze(0).expand(B, Vsz)
            A = ch.unsqueeze(1).contiguous()
            attain = torch.where(live, A[:, 0, :], torch.full_like(logits, float("nan")))
            lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
            hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values
            per_step = torch.where(active, budget / slots.clamp_min(1), lo)
            feas = active & (per_step >= lo - 1e-6) & (per_step <= hi + 1e-6)
            b = per_step.clamp(min=lo, max=hi).unsqueeze(-1)
            res = project(logits, A, b, live)
            q = torch.where(res.converged.unsqueeze(-1) & feas.unsqueeze(-1),
                            res.q, torch.softmax(logits, -1))
            n_infeasible += (active & ~feas).float()
            n_feasible += feas.float()
            n_projected += (feas & res.converged).float()
            n_solver_fallback += (feas & ~res.converged).float()
        else:
            q = torch.softmax(logits, dim=-1)

        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        bad = q.sum(-1) <= 0
        if bool(bad.any()):
            safe = torch.softmax(logits, dim=-1)
            safe = torch.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
            still_bad = bad & (safe.sum(-1) <= 0)
            q = torch.where(bad.unsqueeze(-1), safe, q)
            if bool(still_bad.any()):
                onehot = torch.zeros_like(q)
                onehot[:, eos] = 1.0
                q = torch.where(still_bad.unsqueeze(-1), onehot, q)
            n_degenerate += bad.float()
        nxt = torch.multinomial(q, 1).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
        done = done | (nxt == eos)
        if bool(done.all()):
            break

    model.train()
    seqs = [[int(t) for t in row if True] for row in ids.cpu().numpy()]
    tot = (n_infeasible + n_feasible).clamp_min(1)
    return seqs, {
        "per_step_infeasible_fraction": float((n_infeasible / tot).mean()),
        "mean_steps_projected": float(n_projected.mean()),
        "mean_steps_feasible": float(n_feasible.mean()),
        "mean_steps_solver_fallback": float(n_solver_fallback.mean()),
        "per_step_solver_fallback_fraction": float((n_solver_fallback / tot).mean()),
        "degenerate_row_rate": float((n_degenerate / tot).mean()),
        "sequences_with_any_degenerate_step": float((n_degenerate > 0).float().mean()),
    }
