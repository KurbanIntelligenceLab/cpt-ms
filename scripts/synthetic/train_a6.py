"""A6: fine-tune toward the sequence-level tilt directly, the GDC/DPG comparator.

WHAT THIS COMPARES AND WHY IT IS THE RIGHT COMPARISON. Khalifa, Elsahar and Dymetman
(ICLR 2021) pose controlled generation as moment constraints and derive the same KL-minimal
exponential-family optimum this project calls the tilt, then train an autoregressive model
toward it by distributional policy gradient. That is the sequence-level target the per-step
projection approximates, so "how much does the surrogate lose?" is answerable only against a
model actually trained toward the tilt. Everything before this measured the surrogate against
the tilt's exact distribution; this measures it against a TRAINED APPROXIMATION of the tilt,
which is what a practitioner would deploy.

WHY THE SYNTHETIC TASK IS THE STRONGEST PLACE FOR IT. The task is exactly enumerable, so both
arms' induced distributions are computed rather than sampled and every divergence below is
exact. On crystals the same comparison would rest on sampled estimates of a distribution over
an astronomically large space, which is a weaker measurement of the same thing.

HOW A6 IS TRAINED. The tilt is available in closed form here, so A6 does maximum likelihood on
samples drawn from it. That is the fixed point DPG converges to when its proposal matches the
target: DPG's gradient is an importance-weighted likelihood gradient, and with exact samples
the weights are one. Training on exact tilt samples therefore gives A6 the BEST case of the
method it stands for, rather than a reimplementation whose gradient estimator could be blamed
for the result. A6 gets the mask (Regime S) like every other arm; the difference from A3 is
that A6 has no decode-time projection and A3 has no tilt supervision.

WHAT WOULD FALSIFY THE PAPER'S FRAMING. If A6 reached the tilt closely AND matched A3 on
constraint satisfaction, the projection would be a worse route to the same place and the
contribution would be the cost structure alone. That is the outcome this script is set up to
detect, not to avoid.
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
from scripts.synthetic.targets import kl, tilting_target, tv
from scripts.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "scripts/synthetic/train_a6.py"


def sample_from(p: np.ndarray, task, n: int, rng) -> np.ndarray:
    return task.sequences[rng.choice(len(p), size=n, p=p)]


def main(steps: int | None = None, seeds=None, smoke: bool = False) -> Path:
    from scripts.train.train_a2 import (TinyDecoder, TrainConfig,
                                        induced_distribution_from_model, prefix_masks)

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
        return {
            "mean_charge": float((q * charge).sum()),
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
        tcfg = TrainConfig(arm="A2", steps=steps or (200 if smoke else 4000))
        model = TinyDecoder(cfg.vocab_size, cfg.length, tcfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / tcfg.warmup))
        t0 = time.time()
        for _ in range(1, tcfg.steps + 1):
            # MLE on exact tilt samples: DPG's fixed point with unit importance weights.
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

    fields = ["mean_charge", "kl_from_tilt", "kl_to_tilt", "tv_vs_tilt",
              "kl_from_base", "mass_outside_S", "hard_violation_rate"]

    payload = {
        "arm": "A6: MLE on exact samples from the sequence-level tilt, mask active",
        "why_this_training_rule": (
            "the tilt is closed-form here, so MLE on its exact samples is DPG's fixed point "
            "with unit importance weights -- A6 therefore gets the best case of the method it "
            "stands for, and no gradient-estimator artefact can be blamed for the result"),
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
    return emit("WS7_a6_tilt_comparator" + ("_smoke" if smoke else ""), payload,
                config={"steps": payload["steps"], "seeds": list(seeds),
                        "arm": "A6", "training_rule": "MLE on exact tilt samples"},
                seeds=list(seeds), script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(steps=a.steps, smoke=a.smoke))
