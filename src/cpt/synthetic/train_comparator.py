from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cpt.common.artifacts import emit
from cpt.synthetic.targets import kl, tilting_target, tv
from cpt.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "src/cpt/synthetic/train_comparator.py"


def sample_from(p: np.ndarray, task, n: int, rng) -> np.ndarray:
    return task.sequences[rng.choice(len(p), size=n, p=p)]


def main(steps: int | None = None, seeds=None, smoke: bool = False) -> Path:
    from cpt.train.train_synthetic import (
        TinyDecoder,
        TrainConfig,
        induced_distribution_from_model,
        prefix_masks,
    )

    seeds = seeds or ([0] if smoke else [0, 1, 2])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TaskConfig()
    task = enumerate_task(cfg)

    base = np.where(task.feasible_S, task.p_data, 0.0)
    base = base / base.sum()
    tilt, lam = tilting_target(task, apply_S=True)
    charge = task.total_charge.astype(np.float64)

    def stats(q: np.ndarray) -> dict:
        m = q > 0
        mu = float((q * charge).sum())
        return {
            "mean_charge": mu,
            "charge_variance": float((q * (charge - mu) ** 2).sum()),
            "kl_from_tilt": kl(tilt, q),
            "kl_to_tilt": kl(q, tilt),
            "tv_vs_tilt": tv(q, tilt),
            "kl_from_base": kl(base, q),
            "mass_outside_S": float(q[~task.feasible_S].sum()),
            "hard_violation_rate": float(q[charge != 0.0].sum()),
            "support_size": int(m.sum()),
        }

    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        tcfg = TrainConfig(arm="mask_only", steps=steps or (200 if smoke else 4000))
        model = TinyDecoder(cfg.vocab_size, cfg.length, tcfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / tcfg.warmup))
        t0 = time.time()
        for _ in range(1, tcfg.steps + 1):
            seqs = sample_from(tilt, task, tcfg.batch, rng)
            x = torch.from_numpy(seqs.astype(np.int64)).to(dev)
            inp = torch.cat([torch.full((tcfg.batch, 1), model.bos, device=dev,
                                        dtype=torch.long), x[:, :-1]], dim=1)
            logits = model(inp)
            mk = torch.from_numpy(prefix_masks(task, seqs, cfg)).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))
            logp = torch.log_softmax(logits, dim=-1)
            loss = -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        q = induced_distribution_from_model(model, task, cfg, tcfg, dev,
                                            project_at_inference=False)
        runs.append({"seed": int(seed), "wall_clock_s": round(time.time() - t0, 1),
                     **stats(q)})

    def agg(field: str) -> dict:
        vals = [r[field] for r in runs]
        return {"mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    fields = ["mean_charge", "charge_variance", "kl_from_tilt", "kl_to_tilt", "tv_vs_tilt",
              "kl_from_base", "mass_outside_S", "hard_violation_rate"]

    payload = {
        "arm": "comparator: MLE on exact samples from the sequence-level tilt, mask active",
        "why_this_training_rule": (
            "The comparator fits exact samples from the closed-form sequence tilt by maximum likelihood."),
        "steps": steps or (200 if smoke else 4000),
        "seeds": list(seeds),
        "tilt_multiplier": lam,
        "reference": {
            "tilt": stats(tilt),
            "base_S_conditioned": stats(base),
        },
        "runs": runs,
        "summary": {f: agg(f) for f in fields},
        "exactness_note": ("induced distributions are enumerated over all "
                           f"{len(task.sequences)} sequences, so every divergence is exact "
                           "rather than estimated"),
    }
    return emit("tilt_comparator" + ("_smoke" if smoke else ""), payload,
                config={"steps": payload["steps"], "seeds": list(seeds),
                        "arm": "comparator", "training_rule": "MLE on exact tilt samples"},
                seeds=list(seeds), script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(steps=a.steps, smoke=a.smoke))
