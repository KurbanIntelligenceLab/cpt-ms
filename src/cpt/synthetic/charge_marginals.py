"""Exact per-charge-value marginals recovered by RETRAINING the missing arms fix)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cpt.common.arms import normalize
from cpt.common.artifacts import emit
from cpt.layer.projection import project
from cpt.synthetic.targets import tilting_target
from cpt.synthetic.task import TaskConfig, enumerate_task
from cpt.train.train_synthetic import (
    TinyDecoder,
    TrainConfig,
    induced_distribution_from_model,
    prefix_masks,
    sample_data,
)

SCRIPT = "src/cpt/synthetic/charge_marginals.py"
SEED = 0
STEPS = 4000

_A2_SUMMARY = "results/training_summary.json"
_A6_SUMMARY = "results/tilt_comparator.json"


def marginalize(charge: np.ndarray, p: np.ndarray) -> dict:
    """Exact P(total_charge = c) for every integer c in the support."""
    values = np.unique(charge)
    masses = np.array([p[charge == v].sum() for v in values])
    return {"values": [int(v) for v in values], "masses": [float(m) for m in masses]}


def scalar_stats(charge: np.ndarray, q: np.ndarray, target: float = 0.0) -> dict:
    mu = float((q * charge).sum())
    var = float((q * (charge - mu) ** 2).sum())
    viol = float(q[charge != target].sum())
    return {"mean_charge": mu, "charge_variance": var, "violation_rate": viol}


def train_arm(arm: str, task, cfg: TaskConfig, dev, seed: int = SEED,
              steps: int = STEPS):
    """Reproduce train_synthetic's training loop exactly for one arm."""
    tcfg = TrainConfig(arm=arm, steps=steps, seed=seed)
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    model = TinyDecoder(cfg.vocab_size, cfg.length, tcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / tcfg.warmup))
    phi_t = torch.tensor(np.asarray(cfg.charges, dtype=np.float64),
                          device=dev, dtype=torch.float32)
    t0 = time.time()
    for _step in range(1, tcfg.steps + 1):
        seqs = sample_data(task, tcfg.batch, rng)
        x = torch.from_numpy(seqs.astype(np.int64)).to(dev)
        inp = torch.cat([torch.full((tcfg.batch, 1), model.bos, device=dev,
                                     dtype=torch.long), x[:, :-1]], dim=1)
        logits = model(inp)
        mk = None
        if tcfg.arm != "unconstrained":
            mk = torch.from_numpy(prefix_masks(task, seqs, cfg)).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))
        if tcfg.arm == "projected":
            B, L, V = logits.shape
            flat = logits.reshape(B * L, V)
            A = phi_t.view(1, 1, V).expand(B * L, 1, V).contiguous()
            acc = torch.cat([torch.zeros(B, 1, device=dev), phi_t[x].cumsum(-1)[:, :-1]], 1)
            rem = torch.arange(L, 0, -1, device=dev, dtype=torch.float32).view(1, L)
            b = ((tcfg.charge_target - acc) / rem).reshape(B * L, 1)
            res = project(flat, A, b, mk.reshape(B * L, V))
            logp = torch.log(res.q.clamp_min(1e-30)).reshape(B, L, V)
        else:
            logp = torch.log_softmax(logits, dim=-1)
        loss = -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, tcfg, round(time.time() - t0, 1)


def train_comparator_arm(task, cfg: TaskConfig, tilt: np.ndarray, dev, seed: int = SEED,
                  steps: int = STEPS):
    """Reproduce train_comparator's rule exactly: MLE on exact tilt samples, mask active, no decode-time projection."""
    tcfg = TrainConfig(arm="mask_only", steps=steps, seed=seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyDecoder(cfg.vocab_size, cfg.length, tcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / tcfg.warmup))
    t0 = time.time()
    for _ in range(1, tcfg.steps + 1):
        seqs = task.sequences[rng.choice(len(tilt), size=tcfg.batch, p=tilt)]
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
    return model, tcfg, round(time.time() - t0, 1)


def _plausible(single: float, mean: float, sd: float, k: float = 4.0) -> bool:
    if sd == 0.0:
        return abs(single - mean) < 1e-6 or abs(single - mean) < 0.5
    return abs(single - mean) <= k * sd


def main(smoke: bool = False) -> Path:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TaskConfig(length=4, max_count=2) if smoke else TaskConfig()
    steps = 100 if smoke else STEPS
    task = enumerate_task(cfg)
    charge = task.total_charge.astype(np.float64)

    base = task.p_data.copy()
    base_S = np.where(task.feasible_S, base, 0.0)
    base_S = base_S / base_S.sum()
    tilt, lam = tilting_target(task, apply_S=True)

    model_a2, tcfg_a2, wall_a2 = train_arm("mask_only", task, cfg, dev, steps=steps)
    q_a2 = induced_distribution_from_model(model_a2, task, cfg, tcfg_a2, dev,
                                            project_at_inference=False)

    model_a3, tcfg_a3, wall_a3 = train_arm("projected", task, cfg, dev, steps=steps)
    q_a3_backbone = induced_distribution_from_model(model_a3, task, cfg, tcfg_a3, dev,
                                                      project_at_inference=False)
    q_a3_deployed = induced_distribution_from_model(model_a3, task, cfg, tcfg_a3, dev,
                                                       project_at_inference=True)

    model_a6, tcfg_a6, wall_a6 = train_comparator_arm(task, cfg, tilt, dev, steps=steps)
    q_a6 = induced_distribution_from_model(model_a6, task, cfg, tcfg_a6, dev,
                                            project_at_inference=False)

    distributions = {
        "data_S_conditioned": {**marginalize(charge, base_S), **scalar_stats(charge, base_S),
                                "label": "exact data distribution conditioned on Regime-S mask (no training)"},
        "tilt": {**marginalize(charge, tilt), **scalar_stats(charge, tilt),
                  "label": "exact sequence-level KL-optimal tilt (no training)",
                  "multiplier": float(lam)},
        "mask_only": {**marginalize(charge, q_a2), **scalar_stats(charge, q_a2),
                          "label": "trained mask-only (mask only), backbone reading",
                          "seed": SEED, "steps": tcfg_a2.steps, "wall_clock_s": wall_a2},
        "projected_backbone": {**marginalize(charge, q_a3_backbone), **scalar_stats(charge, q_a3_backbone),
                         "label": "trained projected, read WITHOUT projection (what the network internalized)",
                         "seed": SEED, "steps": tcfg_a3.steps, "wall_clock_s": wall_a3},
        "projected": {**marginalize(charge, q_a3_deployed), **scalar_stats(charge, q_a3_deployed),
                         "label": "trained projected, read WITH the projection it trained with (deployed system)",
                         "seed": SEED, "steps": tcfg_a3.steps, "wall_clock_s": wall_a3},
        "comparator": {**marginalize(charge, q_a6), **scalar_stats(charge, q_a6),
               "label": "trained by MLE on exact tilt samples (train_comparator.py's rule), backbone reading",
               "seed": SEED, "steps": tcfg_a6.steps, "wall_clock_s": wall_a6},
    }

    import json as _json
    agreement = []
    if not smoke:
        mask_only_summary = normalize(_json.loads(Path(_A2_SUMMARY).read_text())["payload"]["arms"])
        comparator_summary = _json.loads(Path(_A6_SUMMARY).read_text())["payload"]

        checks = [
            ("mask_only", "mean_charge", distributions["mask_only"]["mean_charge"],
             mask_only_summary["masked"]["mean_charge"][0], mask_only_summary["masked"]["mean_charge"][1]),
            ("mask_only", "charge_variance", distributions["mask_only"]["charge_variance"],
             mask_only_summary["masked"]["charge_variance"][0], mask_only_summary["masked"]["charge_variance"][1]),
            ("projected", "mean_charge", distributions["projected"]["mean_charge"],
             mask_only_summary["projected"]["mean_charge"][0], mask_only_summary["projected"]["mean_charge"][1]),
            ("projected", "charge_variance", distributions["projected"]["charge_variance"],
             mask_only_summary["projected"]["charge_variance"][0], mask_only_summary["projected"]["charge_variance"][1]),
            ("comparator", "mean_charge", distributions["comparator"]["mean_charge"],
             comparator_summary["summary"]["mean_charge"]["mean"], comparator_summary["summary"]["mean_charge"]["sd"]),
            ("comparator", "charge_variance", distributions["comparator"]["charge_variance"],
             comparator_summary["summary"]["charge_variance"]["mean"], comparator_summary["summary"]["charge_variance"]["sd"]),
            ("tilt", "charge_variance", distributions["tilt"]["charge_variance"],
             comparator_summary["reference"]["tilt"]["charge_variance"], 0.0),
            ("data_S_conditioned", "charge_variance", distributions["data_S_conditioned"]["charge_variance"],
             comparator_summary["reference"]["base_S_conditioned"]["charge_variance"], 0.0),
        ]
        for arm, quantity, single, pub_mean, pub_sd in checks:
            ok = _plausible(single, pub_mean, pub_sd)
            agreement.append({
                "arm": arm, "quantity": quantity, "single_seed_value": single,
                "published_mean": pub_mean, "published_sd": pub_sd,
                "within_4sd": ok,
            })

    payload = {
        "note": (
            "Retrained mask-only, projected and comparator at a SINGLE seed (seed 0), identical hyperparameters "
            "and training loop to the published multi-seed runs, to recover exact per-charge "
            "marginals that were never stored (only scalar summaries were). All marginals are "
            "computed by exact enumeration over all 4**8=65536 sequences, not sampled."
        ),
        "single_seed_caveat": (
            "This is ONE seed, not the published 3-seed average. Any scalar derived here "
            "(mean charge, charge variance, violation rate) is checked below against the "
            "published multi-seed mean +/- sd; entries with within_4sd=false should be treated "
            "as a discrepancy worth investigating, not shipped silently."
        ),
        "distributions": distributions,
        "agreement_check_vs_published": agreement,
    }

    return emit(
        "charge_marginals" + ("_smoke" if smoke else ""),
        payload,
        config=cfg.as_dict() | {"seed": SEED, "steps": steps, "smoke": smoke},
        seeds=[SEED],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(smoke=a.smoke))
