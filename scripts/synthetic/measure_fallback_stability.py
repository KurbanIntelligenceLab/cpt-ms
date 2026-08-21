from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.layer.projection import project
from scripts.synthetic.task import TaskConfig, enumerate_task
from scripts.train.train_a2 import TinyDecoder, TrainConfig, prefix_masks, sample_data

SCRIPT = "scripts/synthetic/measure_fallback_stability.py"


def run(steps: int, charge_targets: list[float], seed: int) -> dict:
    dev = torch.device("cpu")
    cfg = TaskConfig()
    task = enumerate_task(cfg)
    per_target = {}

    for charge_target in charge_targets:
        tcfg = TrainConfig(arm="A3", steps=steps, seed=seed, charge_target=charge_target)
        torch.manual_seed(tcfg.seed)
        rng = np.random.default_rng(tcfg.seed)
        model = TinyDecoder(cfg.vocab_size, cfg.length, tcfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tcfg.warmup))
        phi_t = torch.tensor(np.asarray(cfg.charges, dtype=np.float64), device=dev, dtype=torch.float32)

        losses, grad_norms, infeas_fracs = [], [], []
        t0 = time.time()
        for step in range(1, steps + 1):
            seqs = sample_data(task, tcfg.batch, rng)
            x = torch.from_numpy(seqs.astype(np.int64)).to(dev)
            inp = torch.cat([torch.full((tcfg.batch, 1), model.bos, device=dev, dtype=torch.long),
                             x[:, :-1]], dim=1)
            logits = model(inp)
            mk = torch.from_numpy(prefix_masks(task, seqs, cfg)).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))

            B, L, V = logits.shape
            flat = logits.reshape(B * L, V)
            mkf = mk.reshape(B * L, V)
            A = phi_t.view(1, 1, V).expand(B * L, 1, V).contiguous()
            acc = torch.cat([torch.zeros(B, 1, device=dev), phi_t[x].cumsum(-1)[:, :-1]], 1)
            rem = torch.arange(L, 0, -1, device=dev, dtype=torch.float32).view(1, L)
            b = ((tcfg.charge_target - acc) / rem).reshape(B * L, 1)

            attain = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("nan")))
            lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
            hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values
            bb = b.squeeze(-1)
            infeasible_by_range = (bb < lo - 1e-9) | (bb > hi + 1e-9)

            res = project(flat, A, b, mkf)
            fallback = ~res.converged
            q = torch.where(fallback.unsqueeze(-1), torch.softmax(flat, -1), res.q)
            logp = torch.log(q.clamp_min(1e-30)).reshape(B, L, V)

            loss = -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step()
            sched.step()

            infeas_frac = float(infeasible_by_range.float().mean().item())
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

        per_target[str(charge_target)] = {
            "charge_target": charge_target,
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

    infeas_means = np.array([v["mean_infeasible_fraction_full_run"] for v in per_target.values()])
    loss_var_full = np.array([v["loss_variance_full_run"] for v in per_target.values()])
    grad_var_full = np.array([v["grad_norm_variance_full_run"] for v in per_target.values()])
    loss_var_late = np.array([v["loss_variance_late_window"] for v in per_target.values()])
    grad_var_late = np.array([v["grad_norm_variance_late_window"] for v in per_target.values()])

    def _corr(a, b):
        if a.std() > 0 and b.std() > 0:
            return float(np.corrcoef(a, b)[0, 1])
        return None

    return {
        "per_target": per_target,
        "cross_target_summary": {
            "mean_infeasible_fraction": infeas_means.tolist(),
            "loss_variance_full_run": loss_var_full.tolist(),
            "grad_norm_variance_full_run": grad_var_full.tolist(),
            "loss_variance_late_window": loss_var_late.tolist(),
            "grad_norm_variance_late_window": grad_var_late.tolist(),
        },
        "pearson_r_infeas_vs_loss_variance_full_run": _corr(infeas_means, loss_var_full),
        "pearson_r_infeas_vs_grad_norm_variance_full_run": _corr(infeas_means, grad_var_full),
        "pearson_r_infeas_vs_loss_variance_late_window": _corr(infeas_means, loss_var_late),
        "pearson_r_infeas_vs_grad_norm_variance_late_window": _corr(infeas_means, grad_var_late),
    }


def main(steps: int, charge_targets: list[float], seed: int, smoke: bool) -> Path:
    payload = run(steps=steps, charge_targets=charge_targets, seed=seed)
    config = {"steps": steps, "charge_targets": charge_targets, "seed": seed, "smoke": smoke}
    return emit(
        "synthetic_stability_regrun" + ("_smoke" if smoke else ""),
        payload,
        config=config,
        seeds=[seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--charge-targets", type=float, nargs="+", default=[0.0, 4.0, 6.5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    path = main(steps=args.steps, charge_targets=args.charge_targets, seed=args.seed, smoke=args.smoke)
    print(path)
