from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cpt.common.artifacts import emit
from cpt.crystal.channels import budget_constraint
from cpt.crystal.mp20 import ATOMIC_MASS, ELEMENTS, CrystalVocab, load_split
from cpt.layer.projection import project
from cpt.train.train_crystal import (
    CrystalDecoder,
    CrystalTrainConfig,
    GrammarMask,
    _mass_budget,
    pad_batch,
)

SCRIPT = "src/cpt/crystal/measure_fallback_stability.py"


def run(steps: int, targets: list[float], seed: int, train_limit: int) -> dict:
    dev = torch.device("cpu")
    per_target = {}

    for density_target in targets:
        cfg = CrystalTrainConfig(arm="projected", steps=steps, seed=seed,
                                  train_limit=train_limit, density_target=density_target)
        train = load_split("train", limit=cfg.train_limit)
        V = CrystalVocab(ELEMENTS, order=cfg.order)
        gm = GrammarMask(V, order=cfg.order)
        enc = [V.encode(c) for c in train]
        max_len = max(len(e) for e in enc)
        data = pad_batch(enc, V.index["<EOS>"])

        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        model = CrystalDecoder(len(V), max_len + 2, cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
        mass_t = torch.tensor(
            [ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0 for t in V.tokens],
            device=dev, dtype=torch.float32,
        )

        losses, grad_norms, infeas_fracs = [], [], []
        t0 = time.time()
        for _step in range(1, steps + 1):
            idx = rng.integers(0, len(data), cfg.batch)
            x = torch.from_numpy(data[idx]).to(dev)
            inp, tgt = x[:, :-1], x[:, 1:]
            logits = model(inp)

            mk = torch.from_numpy(gm.masks_for(x.cpu().numpy())[:, 1:, :]).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))

            B, L, Vs = logits.shape
            flat = logits.reshape(B * L, Vs)
            mkf = mk.reshape(B * L, Vs)
            xc = x.cpu().numpy()
            mass = budget_constraint(flat, mkf, xc, V, cfg, dev, channel=mass_t,
                                     budget_fn=lambda p: _mass_budget(p, V, cfg))
            A, b, act = mass.A, mass.b, mass.active

            res = project(flat, A, b, mkf)
            fallback = ~(res.converged & act)
            q = torch.where(fallback.unsqueeze(-1), torch.softmax(flat, -1), res.q)
            logp = torch.log(q.clamp_min(1e-30)).reshape(B, L, Vs)

            loss = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step(); sched.step()

            infeas_frac = float((act & fallback).float().sum().item() / max(int(act.sum().item()), 1))
            losses.append(float(loss.item()))
            grad_norms.append(grad_norm)
            infeas_fracs.append(infeas_frac)

        losses_arr = np.array(losses)
        grad_arr = np.array(grad_norms)
        infeas_arr = np.array(infeas_fracs)

        window = max(steps // 10, 1)
        late_losses = losses_arr[-window:]
        late_grad = grad_arr[-window:]
        late_infeas = infeas_arr[-window:]

        per_target[str(density_target)] = {
            "density_target": density_target,
            "steps": steps,
            "wall_clock_s": round(time.time() - t0, 1),
            "loss_trace": losses,
            "grad_norm_trace": grad_norms,
            "per_step_infeasible_fraction_trace": infeas_fracs,
            "loss_variance_full_run": float(losses_arr.var()),
            "loss_variance_late_window": float(late_losses.var()),
            "grad_norm_mean_full_run": float(grad_arr.mean()),
            "grad_norm_variance_full_run": float(grad_arr.var()),
            "grad_norm_mean_late_window": float(late_grad.mean()),
            "grad_norm_variance_late_window": float(late_grad.var()),
            "mean_infeasible_fraction_full_run": float(infeas_arr.mean()),
            "mean_infeasible_fraction_late_window": float(late_infeas.mean()),
            "late_window_size": window,
        }

    corr_grad_infeas = []
    corr_loss_infeas = []
    for v in per_target.values():
        infeas = np.array(v["per_step_infeasible_fraction_trace"])
        grad = np.array(v["grad_norm_trace"])
        loss = np.array(v["loss_trace"])
        if infeas.std() > 0 and grad.std() > 0:
            corr_grad_infeas.append(float(np.corrcoef(infeas, grad)[0, 1]))
        if infeas.std() > 0 and loss.std() > 0:
            corr_loss_infeas.append(float(np.corrcoef(infeas, loss)[0, 1]))

    return {
        "per_target": per_target,
        "pearson_r_grad_norm_vs_infeasible_fraction_within_run": corr_grad_infeas,
        "pearson_r_loss_vs_infeasible_fraction_within_run": corr_loss_infeas,
        "cross_target_mean_infeasible_fraction_vs_grad_norm_variance": {
            "infeas": [v["mean_infeasible_fraction_full_run"] for v in per_target.values()],
            "grad_norm_variance": [v["grad_norm_variance_full_run"] for v in per_target.values()],
        },
    }


def main(steps: int, targets: list[float], seed: int, train_limit: int, smoke: bool) -> Path:
    payload = run(steps=steps, targets=targets, seed=seed, train_limit=train_limit)
    config = {"steps": steps, "targets": targets, "seed": seed,
              "train_limit": train_limit, "smoke": smoke}
    return emit(
        "stability_regrun" + ("_smoke" if smoke else ""),
        payload,
        config=config,
        seeds=[seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--targets", type=float, nargs="+", default=[5.0, 1.5, 13.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    path = main(steps=args.steps, targets=args.targets, seed=args.seed,
                train_limit=args.train_limit, smoke=args.smoke)
    print(path)
