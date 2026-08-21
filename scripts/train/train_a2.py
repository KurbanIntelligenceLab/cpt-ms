"""A2: masked-softmax baseline trained on the synthetic constrained-sequence task.

A2 is the Regime-S arm — a decoder whose per-step distribution is restricted to the
mask's legal set, with no Regime-E projection. It is the baseline every later claim
is measured against, and it is what CP0's converged-stage reading, the A1-vs-A2
delta, and the CP2 benchmark all wait on.

Trained on the synthetic task rather than crystals deliberately: the synthetic task
has exact enumerable targets, so a trained model's induced distribution can be
compared against the sequence-level optimum computed in closed form. On MP-20 no
such ground truth exists and every comparison would be against another
approximation.

Three arms, so the comparison the project needs is available at the end:

  A1  unconstrained decoder, no mask         — the "does masking help" control
  A2  masked decoder, Regime S only          — the baseline
  A3  masked decoder + per-step projection   — the method, at one active constraint,
                                               which the CP2 benchmark found to be
                                               the affordable configuration

Every arm shares initialization, data order, and optimizer settings, so differences
are attributable to the constraint treatment rather than to training noise.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.layer.projection import project
from scripts.synthetic.targets import kl, tilting_target
from scripts.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "scripts/train/train_a2.py"


@dataclass
class TrainConfig:
    arm: str = "A2"
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
    """Causal decoder over the synthetic vocabulary.

    Deliberately small: the task has a 4-symbol vocabulary at length 8, so capacity
    is not the interesting variable — the constraint treatment is.
    """

    def __init__(self, vocab: int, length: int, cfg: TrainConfig):
        super().__init__()
        self.tok = nn.Embedding(vocab + 1, cfg.d_model)   # +1 for BOS
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
    """Legal-token mask at every position, computed from the prefix alone.

    Regime S is a per-symbol multiplicity budget, which is prefix-closed, so no
    lookahead is needed. That is what makes A2 trainable without a straight-through
    estimator.
    """
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
    """Exact induced distribution over all sequences, by enumeration.

    The task is small enough to score every sequence, so the trained model's
    distribution is obtained exactly rather than estimated by sampling. That removes
    sampling error from every downstream comparison against the closed-form optimum.

    `project_at_inference` distinguishes two genuinely different quantities, and
    conflating them is the easy mistake here:

      False — what the BACKBONE INTERNALIZED. The model is read with the mask only.
              For A3 this answers "did training through the projection teach the
              network the constraint", which is the interesting scientific question
              but is NOT the system anyone would deploy.
      True  — the DEPLOYED SYSTEM: model plus the same projection it trained with,
              applied at generation time. This is what an A3 user would actually run.

    Both are reported for A3. Reporting only the first would understate the system;
    reporting only the second would hide whether the network learned anything.
    """
    model.eval()
    seqs = torch.from_numpy(task.sequences.astype(np.int64)).to(device)
    N, L = seqs.shape
    inp = torch.cat([torch.full((N, 1), model.bos, device=device, dtype=torch.long),
                     seqs[:, :-1]], dim=1)
    logits = model(inp)
    mk = None
    if tcfg.arm != "A1":
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
        # Infeasible-step rate: the budget is unattainable when the required
        # increment lies outside the attainable range of the legal tokens. This is
        # the mechanism behind the residual error, so it is measured rather than
        # inferred.
        lo_att = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("inf"))).min(-1).values
        hi_att = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("-inf"))).max(-1).values
        bb = b.squeeze(-1)
        # Weighted by the sequence's own induced probability, NOT an unweighted mean
        # over the enumeration. The unweighted version is model-independent -- b
        # depends only on the sequence and the attainable range only on the mask --
        # so it cannot be compared against WS1's probability-weighted figure. Bit
        # -identical values across seeds is the signature of that mistake.
        globals()["_LAST_INFEASIBLE_MATRIX"] = (
            (bb < lo_att - 1e-9) | (bb > hi_att + 1e-9)
        ).reshape(N, L).double()
        # No clamp when SCORING. The projection returns exactly 0.0 at masked
        # entries, so an illegal token must score as log(0) = -inf and contribute
        # exactly zero mass. Clamping to a positive floor gives every illegal token
        # a probability of that floor instead, and a sequence with one illegal token
        # then carries ~1e-30 -- which is a representable number, not underflow, and
        # is precisely the residual that appeared in the first deployed reading.
        # It was an artifact of the evaluation, never a property of the layer.
        #
        # The training path keeps its clamp: there the log feeds a loss whose
        # gradient must stay finite, and the targets are drawn from the feasible
        # support so no illegal token is ever scored.
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
        # The Regime-S claim in one number: a masked model puts NO mass outside
        # the feasible set. For A1 this is the control that should be non-zero.
        "infeasible_mass": float(q[~task.feasible_S].sum()),
        "kl_from_data": kl(q, base),
        "kl_from_tilt": kl(q, tilt),
        "violation_rate": float(q[phi != tcfg.charge_target].sum()),
        "entropy_nats": float(-(q[q > 0] * np.log(q[q > 0])).sum()),
    }


def _sampled_detectability(q_exact, q_fast, task, tcfg, *, n_samples, seed) -> dict:
    """Would a reviewer detect the fp32-Hessian route from generated samples?

    Draws n_samples from each deployed distribution and compares the metrics that
    actually get reported — validity, violation rate, mean charge — with a
    two-proportion test on the violation rate. A null result here is only
    meaningful with the sample size stated, so it is recorded.
    """
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
    """Backbone reading always; deployed reading additionally for A3."""
    m = _metrics(induced_distribution_from_model(model, task, cfg, tcfg, device),
                 task, tcfg, base, tilt)
    if tcfg.arm == "A3":
        qd = induced_distribution_from_model(model, task, cfg, tcfg, device,
                                             project_at_inference=True)
        m["deployed"] = _metrics(qd, task, tcfg, base, tilt)
        # Train/inference precision mismatch: the fp32-Hessian route would deploy a
        # distribution perturbed inside the tolerance ball relative to the one the
        # model trained against. This project's thesis is that train/inference
        # mismatch is the problem worth solving, so shipping one inside the solver
        # needs a measurement, not an argument. Both deployed distributions are
        # computed from the SAME trained weights, so any difference is the
        # precision, not training noise.
        qf = induced_distribution_from_model(model, task, cfg, tcfg, device,
                                             project_at_inference=True,
                                             fast_hessian=True)
        m["deployed_fast_hessian"] = _metrics(qf, task, tcfg, base, tilt)
        # Direct sampled-metric test, not only the distributional bound: draw from
        # both deployed distributions and test the metrics a reviewer would report.
        # The distributions are exact, so sampling error is the ONLY source of
        # difference beyond the precision effect.
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
            w = torch.from_numpy(qd).to(M.device)          # induced sequence mass
            m["deployed"]["per_step_infeasible_fraction"] = float(
                (M.mean(dim=1) * w).sum()
            )
            m["deployed"]["per_step_infeasible_fraction_unweighted"] = float(M.mean())
    return m


def main(smoke: bool = False, arm: str = "A2", steps: int | None = None) -> Path:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = TrainConfig(arm=arm, steps=steps or (200 if smoke else 4000))
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
        if tcfg.arm != "A1":
            mk = torch.from_numpy(prefix_masks(task, seqs, cfg)).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))

        if tcfg.arm == "A3":
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
        f"train_{tcfg.arm}" + ("_smoke" if smoke else ""),
        payload,
        config={"arm": tcfg.arm, "steps": tcfg.steps, "smoke": smoke},
        seeds=[tcfg.seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", default="A2", choices=["A1", "A2", "A3"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    print(main(smoke=a.smoke, arm=a.arm, steps=a.steps))
