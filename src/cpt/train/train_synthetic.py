from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cpt.common.arms import resolve
from cpt.common.artifacts import emit
from cpt.layer.projection import project
from cpt.synthetic.targets import kl, tilting_target
from cpt.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "src/cpt/train/train_synthetic.py"


@dataclass
class TrainConfig:
    arm: str = "mask_only"
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    steps: int = 4000
    batch: int = 256
    lr: float = 3e-4
    warmup: int = 200
    seed: int = 0
    eval_every: int = 500
    charge_target: float = 0.0


class TinyDecoder(nn.Module):
    def __init__(self, vocab: int, length: int, cfg: TrainConfig):
        super().__init__()
        self.tok = nn.Embedding(vocab + 1, cfg.d_model)
        self.pos = nn.Embedding(length, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_head, 4 * cfg.d_model,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layer)
        self.head = nn.Linear(cfg.d_model, vocab)
        self.bos = vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        m = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=m, is_causal=True)
        return self.head(h)


def sample_data(task, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw sequences from the data distribution restricted to S-feasible support."""
    p = np.where(task.feasible_S, task.p_data, 0.0)
    p = p / p.sum()
    return task.sequences[rng.choice(len(p), size=n, p=p)]


def prefix_masks(task, seqs: np.ndarray, cfg: TaskConfig) -> np.ndarray:
    """Legal-token mask at every position, computed from the prefix alone."""
    n, L = seqs.shape
    V = cfg.vocab_size
    out = np.ones((n, L, V), dtype=bool)
    for i in range(n):
        counts = np.zeros(V, dtype=int)
        for t in range(L):
            out[i, t] = counts < cfg.max_count
            counts[seqs[i, t]] += 1
    return out


@torch.no_grad()
def induced_distribution_from_model(
    model, task, cfg, tcfg, device, *, project_at_inference: bool = False,
    fast_hessian: bool = False,
) -> np.ndarray:
    """Exact induced distribution over all sequences, by enumeration."""
    model.eval()
    seqs = torch.from_numpy(task.sequences.astype(np.int64)).to(device)
    N, L = seqs.shape
    inp = torch.cat([torch.full((N, 1), model.bos, device=device, dtype=torch.long),
                     seqs[:, :-1]], dim=1)
    logits = model(inp)
    mk = None
    if tcfg.arm != "unconstrained":
        mk = torch.from_numpy(prefix_masks(task, task.sequences, cfg)).to(device)
        logits = logits.masked_fill(~mk, float("-inf"))

    if project_at_inference and mk is not None:
        phi_t = torch.tensor(np.asarray(cfg.charges, dtype=np.float64),
                             device=device, dtype=torch.float32)
        V = logits.shape[-1]
        flat = logits.reshape(N * L, V)
        A = phi_t.view(1, 1, V).expand(N * L, 1, V).contiguous()
        acc = torch.cat([torch.zeros(N, 1, device=device),
                         phi_t[seqs].cumsum(-1)[:, :-1]], dim=1)
        rem = torch.arange(L, 0, -1, device=device, dtype=torch.float32).view(1, L)
        b = ((tcfg.charge_target - acc) / rem).reshape(N * L, 1)
        mkf = mk.reshape(N * L, V)
        res = project(flat, A, b, mkf,
                      fast_hessian_dtype=(torch.float32 if fast_hessian else None))
        lo_att = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("inf"))).min(-1).values
        hi_att = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("-inf"))).max(-1).values
        bb = b.squeeze(-1)
        globals()["_LAST_INFEASIBLE_MATRIX"] = (
            (bb < lo_att - 1e-9) | (bb > hi_att + 1e-9)
        ).reshape(N, L).double()
        logp = torch.log(res.q).reshape(N, L, V)
    else:
        logp = torch.log_softmax(logits, dim=-1)

    lp = logp.gather(-1, seqs.unsqueeze(-1)).squeeze(-1).sum(-1)
    p = lp.double().exp()
    model.train()
    return (p / p.sum()).cpu().numpy()


def _metrics(q, task, tcfg, base, tilt) -> dict:
    phi = task.total_charge.astype(np.float64)
    mean = float(q @ phi)
    return {
        "mean_charge": mean,
        "charge_error": mean - tcfg.charge_target,
        "charge_variance": float(q @ (phi - mean) ** 2),
        "infeasible_mass": float(q[~task.feasible_S].sum()),
        "kl_from_data": kl(q, base),
        "kl_from_tilt": kl(q, tilt),
        "violation_rate": float(q[phi != tcfg.charge_target].sum()),
        "entropy_nats": float(-(q[q > 0] * np.log(q[q > 0])).sum()),
    }


def _sampled_detectability(q_exact, q_fast, task, tcfg, *, n_samples, seed) -> dict:
    rng = np.random.default_rng(seed)
    phi = task.total_charge.astype(np.float64)
    ie = rng.choice(len(q_exact), size=n_samples, p=q_exact / q_exact.sum())
    iff = rng.choice(len(q_fast), size=n_samples, p=q_fast / q_fast.sum())
    viol_e = float((phi[ie] != tcfg.charge_target).mean())
    viol_f = float((phi[iff] != tcfg.charge_target).mean())
    pb = (viol_e + viol_f) / 2
    se = np.sqrt(2 * pb * (1 - pb) / n_samples) if 0 < pb < 1 else np.inf
    z = abs(viol_e - viol_f) / se if se > 0 else 0.0
    return {
        "n_samples": n_samples,
        "violation_rate_exact": viol_e,
        "violation_rate_fast": viol_f,
        "violation_rate_abs_diff": abs(viol_e - viol_f),
        "two_proportion_z": float(z),
        "detectable_at_0_05": bool(z > 1.959964),
        "mean_charge_exact": float(phi[ie].mean()),
        "mean_charge_fast": float(phi[iff].mean()),
        "infeasible_count_exact": int((~task.feasible_S)[ie].sum()),
        "infeasible_count_fast": int((~task.feasible_S)[iff].sum()),
    }


def evaluate(model, task, cfg, tcfg, device, base, tilt) -> dict:
    """Backbone reading always; deployed reading additionally for projected."""
    m = _metrics(induced_distribution_from_model(model, task, cfg, tcfg, device),
                 task, tcfg, base, tilt)
    if tcfg.arm == "projected":
        qd = induced_distribution_from_model(model, task, cfg, tcfg, device,
                                             project_at_inference=True)
        m["deployed"] = _metrics(qd, task, tcfg, base, tilt)
        qf = induced_distribution_from_model(model, task, cfg, tcfg, device,
                                             project_at_inference=True,
                                             fast_hessian=True)
        m["deployed_fast_hessian"] = _metrics(qf, task, tcfg, base, tilt)
        m["precision_mismatch_sampled"] = _sampled_detectability(
            qd, qf, task, tcfg, n_samples=1_000_000, seed=tcfg.seed)
        m["precision_mismatch"] = {
            "total_variation": float(0.5 * np.abs(qf - qd).sum()),
            "max_abs_prob_diff": float(np.abs(qf - qd).max()),
            "mean_charge_diff": float((qf @ task.total_charge.astype(np.float64))
                                      - (qd @ task.total_charge.astype(np.float64))),
            "infeasible_mass_diff": float(qf[~task.feasible_S].sum()
                                          - qd[~task.feasible_S].sum()),
        }
        M = globals().get("_LAST_INFEASIBLE_MATRIX")
        if M is not None:
            w = torch.from_numpy(qd).to(M.device)
            m["deployed"]["per_step_infeasible_fraction"] = float(
                (M.mean(dim=1) * w).sum()
            )
            m["deployed"]["per_step_infeasible_fraction_unweighted"] = float(M.mean())
    return m


def main(smoke: bool = False, arm: str = "mask_only", steps: int | None = None, seed: int = 0) -> Path:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = TrainConfig(arm=arm, steps=steps or (200 if smoke else 4000), seed=seed)
    if smoke:
        tcfg.eval_every = 100
    cfg = TaskConfig()
    task = enumerate_task(cfg)

    base = np.where(task.feasible_S, task.p_data, 0.0)
    base = base / base.sum()
    tilt, _ = tilting_target(task, apply_S=True)

    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    model = TinyDecoder(cfg.vocab_size, cfg.length, tcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / tcfg.warmup))

    phi_t = torch.tensor(np.asarray(cfg.charges, dtype=np.float64),
                         device=dev, dtype=torch.float32)
    history, t0 = [], time.time()

    for step in range(1, tcfg.steps + 1):
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

        if step % tcfg.eval_every == 0 or step == tcfg.steps:
            m = evaluate(model, task, cfg, tcfg, dev, base, tilt)
            m |= {"step": step, "loss": float(loss.item()),
                  "elapsed_s": round(time.time() - t0, 1)}
            history.append(m)
            print(f"  step {step:5d} loss {m['loss']:.4f} KL_data {m['kl_from_data']:.4f} "
                  f"mean {m['mean_charge']:+.4f} infeasible {m['infeasible_mass']:.3e}",
                  flush=True)

    payload = {
        "arm": tcfg.arm,
        "train_config": asdict(tcfg),
        "task_config": asdict(cfg),
        "device": str(dev),
        "gpu": (torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "final": history[-1],
        "reference": {
            "tilt_kl_from_data": kl(tilt, base),
            "tilt_mean_charge": float(tilt @ task.total_charge.astype(np.float64)),
            "data_mean_charge": float(base @ task.total_charge.astype(np.float64)),
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    return emit(
        f"train_{tcfg.arm}_seed{tcfg.seed}" + ("_smoke" if smoke else ""),
        payload,
        config={"training": asdict(tcfg), "task": asdict(cfg), "smoke": smoke},
        seeds=[tcfg.seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", default="mask_only", type=resolve,
                    choices=["unconstrained", "mask_only", "projected"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(main(smoke=a.smoke, arm=a.arm, steps=a.steps, seed=a.seed))
