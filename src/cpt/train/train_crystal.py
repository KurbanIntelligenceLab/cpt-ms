from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cpt.common.arms import resolve
from cpt.common.artifacts import emit, file_sha256
from cpt.crystal.channels import (
    _additive_budget,
    _element_position,
    _mass_budget,
    budget_constraint,
    mass_channel,
    weight_channel,
)
from cpt.crystal.metrics import _coverage, structural_metrics
from cpt.crystal.model import CrystalDecoder, CrystalTrainConfig, GrammarMask
from cpt.crystal.mp20 import ATOMIC_MASS, ELEMENTS, VALENCE, CrystalVocab, load_split
from cpt.crystal.sampling import pad_batch, sample
from cpt.layer.projection import project

SCRIPT = "src/cpt/train/train_crystal.py"


__all__ = [
    "ATOMIC_MASS",
    "ELEMENTS",
    "VALENCE",
    "CrystalDecoder",
    "CrystalTrainConfig",
    "CrystalVocab",
    "GrammarMask",
    "_additive_budget",
    "_coverage",
    "_element_position",
    "_mass_budget",
    "load_split",
    "main",
    "mass_channel",
    "pad_batch",
    "sample",
    "structural_metrics",
    "weight_channel",
]


def main(smoke: bool = False, arm: str = "mask_only", steps: int | None = None,
         density_target: float | None = None, seed: int | None = None,
         ckpt: str | Path | None = None, valence_target: float | None = None,
         name: str | None = None) -> Path:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CrystalTrainConfig(arm=arm, steps=steps or (300 if smoke else 6000))
    if density_target is not None:
        cfg.density_target = float(density_target)
    if valence_target is not None:
        cfg.valence_target = float(valence_target)
    if seed is not None:
        cfg.seed = int(seed)
    if smoke:
        cfg.eval_every, cfg.n_eval_samples, cfg.train_limit = 150, 512, 2000

    train = load_split("train", limit=cfg.train_limit)
    full_train = list(train)
    full_density = np.array([c.density for c in train])
    n_full = len(train)
    if cfg.arm == "conditional":
        train = [c for c in train
                 if abs(c.density - cfg.density_target) <= cfg.filter_tol]
        if len(train) < 20:
            raise ValueError(
                f"conditional filter at target {cfg.density_target} +/- {cfg.filter_tol} left "
                f"{len(train)} structures of {n_full}; too few to train on"
            )
    V = CrystalVocab(ELEMENTS, order=cfg.order)
    gm = GrammarMask(V, order=cfg.order)
    enc = [V.encode(c) for c in train]
    max_len = max(len(e) for e in enc)
    data = pad_batch(enc, V.index["<EOS>"])
    data_density = full_density
    _full_el: Counter = Counter()
    for c in full_train:
        for e, k in c.formula.items():
            _full_el[e] += k
    _tot_el = sum(_full_el.values())
    full_element_freq = {e: k / _tot_el for e, k in _full_el.items()}
    train_formulas = {tuple(sorted(c.formula.items())) for c in train}
    COV = dict(train_formulas=train_formulas, full_element_freq=full_element_freq)

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = CrystalDecoder(len(V), max_len + 2, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    mass_t = torch.tensor([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                           for t in V.tokens], device=dev, dtype=torch.float32)

    init_ref = None
    if cfg.arm == "projected":
        s0, d0 = sample(model, V, gm, cfg, dev, cfg.n_eval_samples, use_mask=True,
                        project_at_inference=True, max_len=max_len + 2, rng=rng)
        init_ref = structural_metrics(s0, V, gm, cfg, **COV) | d0 | {"step": 0}
        print(f"  step     0 (init) infeasible {d0['per_step_infeasible_fraction']:.4f} "
              f"density {init_ref['mean_density'] if init_ref['mean_density'] else float('nan'):.3f}",
              flush=True)

    history, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        idx = rng.integers(0, len(data), cfg.batch)
        x = torch.from_numpy(data[idx]).to(dev)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)

        if cfg.arm != "unconstrained":
            mk = torch.from_numpy(gm.masks_for(x.cpu().numpy())[:, 1:, :]).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))

        if cfg.arm == "projected":
            B, L, Vs = logits.shape
            flat = logits.reshape(B * L, Vs)
            mkf = mk.reshape(B * L, Vs)
            xc = x.cpu().numpy()
            mass = budget_constraint(
                flat, mkf, xc, V, cfg, dev,
                channel=((lambda p: mass_channel(p, V)) if cfg.channel == "prefix" else mass_t),
                budget_fn=lambda p: _mass_budget(p, V, cfg))
            A, b, act = mass.A, mass.b, mass.active

            res = project(flat, A, b, mkf)
            q = torch.where((res.converged & act).unsqueeze(-1), res.q,
                            torch.softmax(flat, -1))
            logp = torch.log(q.clamp_min(1e-30)).reshape(B, L, Vs)
        else:
            logp = torch.log_softmax(logits, dim=-1)

        loss = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            seqs, diag = sample(model, V, gm, cfg, dev, cfg.n_eval_samples,
                                use_mask=(cfg.arm != "unconstrained"), project_at_inference=False,
                                max_len=max_len + 2, rng=rng)
            m = structural_metrics(seqs, V, gm, cfg, **COV)
            m |= {"step": step, "loss": float(loss.item()),
                  "elapsed_s": round(time.time() - t0, 1)} | diag
            if cfg.arm == "projected":
                sd, dg = sample(model, V, gm, cfg, dev, cfg.n_eval_samples,
                                use_mask=True, project_at_inference=True,
                                max_len=max_len + 2, rng=rng)
                m["deployed"] = structural_metrics(sd, V, gm, cfg, **COV) | dg
                if m["mean_density"] is not None and m["deployed"]["mean_density"] is not None:
                    m["projection_pull"] = m["mean_density"] - m["deployed"]["mean_density"]
                    m["backbone_offset_vs_data"] = m["mean_density"] - float(data_density.mean())
            history.append(m)
            print(f"  step {step:5d} loss {m['loss']:.4f} valid {m['structural_validity_rate']:.4f} "
                  f"density {m['mean_density'] if m['mean_density'] else float('nan'):.3f}",
                  flush=True)

    payload = {
        "arm": cfg.arm,
        "train_config": asdict(cfg),
        "device": str(dev),
        "gpu": (torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(V),
        "n_train_structures": len(train),
        "max_len": max_len,
        "training_set": {
            "n_structures": len(train), "n_full_split": n_full,
            "fraction_of_split": len(train) / n_full,
            "filtered": cfg.arm == "conditional",
            "filter_tol": (cfg.filter_tol if cfg.arm == "conditional" else None),
            "train_density_mean": float(np.mean([c.density for c in train])),
        },
        "data_density": {
            "mean": float(data_density.mean()), "median": float(np.median(data_density)),
            "sd": float(data_density.std()),
        },
        "history": history,
        "final": history[-1],
        "init_reference": init_ref,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    if ckpt is not None:
        ckpt = Path(ckpt)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "train_config": asdict(cfg),
                "trained_density_target": cfg.density_target,
                "max_len": max_len,
                "vocab_order": cfg.order,
                "vocab_size": len(V),
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "elements": list(ELEMENTS),
                "schema_version": 2,
                "tokenizer": V.configuration(),
            },
            ckpt,
        )
        payload["checkpoint"] = {
            "path": ckpt.name,
            "trained_density_target": cfg.density_target,
            "max_len": max_len,
            "vocab_order": cfg.order,
            "bytes": ckpt.stat().st_size,
            "sha256": file_sha256(ckpt),
            "tokenizer": V.configuration(),
        }
    return emit(
        name or f"crystal_{cfg.arm}_density{cfg.density_target:g}_seed{cfg.seed}" + ("_smoke" if smoke else ""),
        payload,
        config={"training": asdict(cfg), "smoke": smoke},
        seeds=[cfg.seed], script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", default="mask_only", type=resolve,
                    choices=["unconstrained", "mask_only", "projected", "conditional"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--density-target", type=float, default=None)
    ap.add_argument("--valence-target", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None,
                    help="output record name; defaults to model, density and seed")
    ap.add_argument("--ckpt", default=None,
                    help="path to save model weights for later zero-shot retargeting")
    a = ap.parse_args()
    print(main(smoke=a.smoke, arm=a.arm, steps=a.steps,
               density_target=a.density_target, seed=a.seed, ckpt=a.ckpt,
               valence_target=a.valence_target, name=a.name))
