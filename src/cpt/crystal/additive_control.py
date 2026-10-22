from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cpt.common.artifacts import emit
from cpt.crystal.channels import budget_constraint
from cpt.crystal.mp20 import ATOMIC_MASS, ELEMENTS, load_split
from cpt.layer.projection import project
from cpt.train.train_crystal import (
    CrystalDecoder,
    CrystalTrainConfig,
    CrystalVocab,
    GrammarMask,
    _mass_budget,
    pad_batch,
    sample,
    structural_metrics,
)

SCRIPT = "src/cpt/crystal/additive_control.py"


def run_density_arm(mode: str, cfg, train, V, gm, max_len, data, dev,
                    train_formulas, full_element_freq, rng, torch_seed) -> dict:
    torch.manual_seed(torch_seed)
    model = CrystalDecoder(len(V), max_len + 2, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    mass_t = torch.tensor([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                           for t in V.tokens], device=dev, dtype=torch.float32)

    t0 = time.time()
    for _step in range(1, cfg.steps + 1):
        idx = rng.integers(0, len(data), cfg.batch)
        x = torch.from_numpy(data[idx]).to(dev)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        mk = torch.from_numpy(gm.masks_for(x.cpu().numpy())[:, 1:, :]).to(dev)
        logits = logits.masked_fill(~mk, float("-inf"))

        if mode == "projected":
            B, L, Vs = logits.shape
            flat = logits.reshape(B * L, Vs)
            mkf = mk.reshape(B * L, Vs)
            xc = x.cpu().numpy()
            mass = budget_constraint(flat, mkf, xc, V, cfg, dev, channel=mass_t,
                                     budget_fn=lambda p: _mass_budget(p, V, cfg))
            A, b, act = mass.A, mass.b, mass.active
            res = project(flat, A, b, mkf)
            q = torch.where((res.converged & act).unsqueeze(-1), res.q, torch.softmax(flat, -1))
            logp = torch.log(q.clamp_min(1e-30)).reshape(B, L, Vs)
        else:
            logp = torch.log_softmax(logits, dim=-1)

        loss = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    seqs, diag = sample(model, V, gm, cfg, dev, cfg.n_eval_samples, use_mask=True,
                        project_at_inference=(mode == "projected"),
                        max_len=max_len + 2, rng=rng)
    m = structural_metrics(seqs, V, gm, cfg, train_formulas=train_formulas,
                           full_element_freq=full_element_freq)
    m |= diag
    m["wall_clock_s"] = round(time.time() - t0, 1)
    m["mode"] = mode
    return m


def main() -> Path:
    cfg = CrystalTrainConfig(arm="projected", steps=600, batch=128, n_eval_samples=512,
                             train_limit=3000, density_target=5.0, order="cell_first")
    dev = torch.device("cpu")

    train = load_split("train", limit=cfg.train_limit)
    V = CrystalVocab(ELEMENTS, order=cfg.order)
    gm = GrammarMask(V, order=cfg.order)
    enc = [V.encode(c) for c in train]
    max_len = max(len(e) for e in enc)
    data = pad_batch(enc, V.index["<EOS>"])

    from collections import Counter
    full_el: Counter = Counter()
    for c in train:
        for e, k in c.formula.items():
            full_el[e] += k
    tot_el = sum(full_el.values())
    full_element_freq = {e: k / tot_el for e, k in full_el.items()}
    train_formulas = {tuple(sorted(c.formula.items())) for c in train}

    results = {}
    for mode, ts in (("unsteered", cfg.seed), ("projected", cfg.seed + 1)):
        rng = np.random.default_rng(ts)
        print(f"running arm {mode}", flush=True)
        results[mode] = run_density_arm(mode, cfg, train, V, gm, max_len, data, dev,
                                        train_formulas, full_element_freq, rng, ts)
        print(f"  {mode}: {results[mode]}", flush=True)

    payload = {
        "property": "mass_density",
        "note": (
            "matched-budget additive-property control: same steps, batch, "
            "train_limit, n_eval_samples and vocabulary as the electronegativity arms, "
            "but with the fixed per-element mass weight table (the mechanism used for "
            "the main-text density results) rather than a learned phi"
        ),
        "config": asdict(cfg),
        "arms": results,
    }
    return emit("additive_control", payload,
               config={"steps": cfg.steps, "train_limit": cfg.train_limit}, seeds=[cfg.seed],
               script=SCRIPT)


if __name__ == "__main__":
    print(main())

