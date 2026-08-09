"""Crystal A2: the three arms on MP-20 with a real backbone (WS-3).

The synthetic task established the mechanism against exact enumerable targets. This
carries it to MP-20, where the targets are approximate and the backbone is a real
model trained on real structures. Two conventions carry over unchanged, per the
review:

  TWO READINGS FOR A3. The backbone reading applies the mask only and reports what
  the network internalized; the deployed reading applies the same projection the
  model trained with and is the system anyone would run. Reporting either alone
  misleads in opposite directions, and the gap between them IS the internalization
  result.

  PROBABILITY-WEIGHTED INFEASIBLE-STEP FRACTION. An unweighted mean over a fixed
  evaluation set is model-independent and therefore not comparable across arms.

What necessarily differs from the synthetic task: the sequence space is not
enumerable, so every distributional quantity is estimated from samples rather than
computed exactly. Sample sizes are stated with each figure and the same draw count
is used for every arm.

The steered quantity is DENSITY, primary on all backbones per
`preregistration/WS2_dial_assignment.md`.
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
from scripts.crystal.mp20 import ATOMIC_MASS, ELEMENTS, CrystalVocab, load_split
from scripts.layer.projection import project

SCRIPT = "scripts/train/train_crystal.py"


@dataclass
class CrystalTrainConfig:
    arm: str = "A2"
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    steps: int = 6000
    batch: int = 128
    lr: float = 3e-4
    warmup: int = 300
    seed: int = 0
    eval_every: int = 1000
    n_eval_samples: int = 4096
    density_target: float = 5.0     # g/cm^3, set from the data below the median
    train_limit: int | None = None
    order: str = "cell_first"      # density needs the volume before the composition
    expected_elements: float = 3.0  # mean distinct elements per MP-20 structure


class CrystalDecoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, cfg: CrystalTrainConfig):
        super().__init__()
        self.tok = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_head, 4 * cfg.d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layer)
        self.head = nn.Linear(cfg.d_model, vocab)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        m = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.head(self.blocks(h, mask=m, is_causal=True))


class GrammarMask:
    """Regime S for the crystal serialization: which tokens are legal next.

    Prefix-closed by construction — the legal set depends only on which block is
    open and what has been emitted — which is what makes the mask trainable without
    a straight-through estimator, exactly as on the synthetic task.
    """

    def __init__(self, V: CrystalVocab, max_elements: int = 6, order: str = "cell_first"):
        self.V = V
        self.order = order
        self.max_elements = max_elements
        self.el = np.array([t.startswith("el:") for t in V.tokens])
        self.cnt = np.array([t.startswith("cnt:") for t in V.tokens])
        self.len_ = np.array([t.startswith("len:") for t in V.tokens])
        self.ang = np.array([t.startswith("ang:") for t in V.tokens])
        self.i = {t: V.index[t] for t in
                  ("<BOS>","<COMP>","<END_COMP>","<CELL>","<END_CELL>","<EOS>")}

    def masks_for(self, ids: np.ndarray) -> np.ndarray:
        """Legal-token mask at every position of every sequence in the batch.

        `out[r, t]` is the set of tokens legal AT position t, i.e. after consuming
        ids[r, :t]. So the state must be advanced by token t-1 before writing row t.
        Writing the mask before consuming the current token instead describes what
        may follow position t-1, which rejects the data's own sequences.
        """
        n, L = ids.shape
        out = np.zeros((n, L, len(self.V)), dtype=bool)
        for r in range(n):
            state, n_el, n_len, n_ang, seen = "bos", 0, 0, 0, set()
            for t in range(L):
                if t > 0:
                    tok = self.V.tokens[ids[r, t - 1]]
                    if tok == "<COMP>":
                        state = "want_el"
                    elif tok.startswith("el:"):
                        seen.add(tok[3:]); state = "want_cnt"
                    elif tok.startswith("cnt:"):
                        n_el += 1; state = "want_el"
                    elif tok == "<END_COMP>":
                        state = "eos" if self.order == "cell_first" else "want_cell"
                    elif tok == "<CELL>":
                        state = "want_len"
                    elif tok.startswith("len:"):
                        n_len += 1; state = "want_len" if n_len < 3 else "want_ang"
                    elif tok.startswith("ang:"):
                        n_ang += 1; state = "want_ang" if n_ang < 3 else "want_end_cell"
                    elif tok == "<END_CELL>":
                        state = "want_comp" if self.order == "cell_first" else "eos"
                m = out[r, t]
                if state == "bos":
                    m[self.i["<CELL>" if self.order == "cell_first" else "<COMP>"]] = True
                elif state == "want_el":
                    m[self.el] = True
                    for e in seen:
                        m[self.V.index[f"el:{e}"]] = False       # no repeats
                    if n_el >= 1:
                        m[self.i["<END_COMP>"]] = True
                    if n_el >= self.max_elements:
                        m[self.el] = False
                        m[self.i["<END_COMP>"]] = True
                elif state == "want_cnt":
                    m[self.cnt] = True
                elif state == "want_cell":
                    m[self.i["<CELL>"]] = True
                elif state == "want_comp":
                    m[self.i["<COMP>"]] = True
                elif state == "want_len":
                    m[self.len_] = True
                elif state == "want_ang":
                    m[self.ang] = True
                elif state == "want_end_cell":
                    m[self.i["<END_CELL>"]] = True
                else:
                    m[self.i["<EOS>"]] = True
        return out


def mass_channel(ids: np.ndarray, V: CrystalVocab) -> np.ndarray:
    """Per-token mass contribution GIVEN the prefix. Shape (n, vocab).

    The constraint channel is prefix-dependent here, which the synthetic task did
    not require. An element token `el:E` contributes nothing on its own — the mass
    it adds is decided by the count token that follows, as ATOMIC_MASS[E] * n. So
    at a count position with pending element E the channel value of `cnt:n` is
    ATOMIC_MASS[E] * n, and at an element position the channel is the element's own
    mass as a proxy for the mass about to be committed.

    A fixed per-token channel makes every count position carry mass zero, so the
    attainable range collapses to {0} and any nonzero budget is infeasible there.
    That is a property of the serialization, not of the constraint.
    """
    n = ids.shape[0]
    out = np.zeros((n, len(V)), dtype=np.float32)
    cnt_idx = [(i, int(t[4:])) for i, t in enumerate(V.tokens) if t.startswith("cnt:")]
    el_mass = np.array([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                        for t in V.tokens], dtype=np.float32)
    for r in range(n):
        pending = None
        for t in range(ids.shape[1]):
            tok = V.tokens[ids[r, t]]
            if tok.startswith("el:"):
                pending = tok[3:]
            elif tok.startswith("cnt:"):
                pending = None
        if pending is not None:
            m = ATOMIC_MASS.get(pending, 0.0)
            for i, k in cnt_idx:
                out[r, i] = m * k
        else:
            out[r] = el_mass
    return out


def _mass_budget(ids: np.ndarray, V: CrystalVocab, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Remaining mass to reach the target density, and remaining element slots.

    Returns (budget_u, slots). Budget is NaN until the cell block closes, because
    until then the volume — and therefore the total mass a target density implies —
    is not determined. That is the actionability boundary WS-2 measured, appearing
    here as an explicit precondition rather than as a silent wrong answer.

    Slots is the number of element tokens still expected. It cannot be known exactly
    (the model decides how many elements to emit), so it is estimated from the mean
    element count in the data. An estimate is sound here because the budget is
    re-derived at every step from the mass actually emitted so far: an over- or
    under-estimate corrects itself on the next token rather than accumulating.
    """
    n, L = ids.shape
    budget = np.full(n, np.nan, dtype=np.float64)
    slots = np.zeros(n, dtype=np.float64)
    for r in range(n):
        lens, angs, mass, n_el, cell_closed = [], [], 0.0, 0, False
        cur_el = None
        for t in range(L):
            tok = V.tokens[ids[r, t]]
            if tok.startswith("len:"):
                j = int(tok[4:]); lens.append(2.0 + (j + 0.5) * 28.0 / V.N_LEN)
            elif tok.startswith("ang:"):
                j = int(tok[4:]); angs.append(20.0 + (j + 0.5) * 140.0 / V.N_ANG)
            elif tok == "<END_CELL>":
                cell_closed = len(lens) == 3 and len(angs) == 3
            elif tok.startswith("el:"):
                cur_el = tok[3:]
            elif tok.startswith("cnt:") and cur_el:
                mass += ATOMIC_MASS.get(cur_el, 0.0) * int(tok[4:])
                n_el += 1; cur_el = None
        if cell_closed:
            vol = V.cell_volume(lens, angs)
            if vol:
                total = cfg.density_target * vol / 1.66053906660
                budget[r] = max(total - mass, 0.0)
                slots[r] = max(cfg.expected_elements - n_el, 1)
    return budget, slots


def pad_batch(seqs: list[list[int]], pad: int) -> np.ndarray:
    L = max(len(s) for s in seqs)
    a = np.full((len(seqs), L), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return a


@torch.no_grad()
def sample(model, V, gm, cfg, device, n, *, use_mask: bool, project_at_inference: bool,
           max_len: int, rng) -> tuple[list[list[int]], dict]:
    """Autoregressive generation. Returns token sequences and diagnostics."""
    model.eval()
    ids = torch.full((n, 1), V.index["<BOS>"], dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    n_infeasible = torch.zeros(n, device=device)
    n_projected = torch.zeros(n, device=device)
    eos = V.index["<EOS>"]
    mass_t = torch.tensor([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                           for t in V.tokens], device=device, dtype=torch.float32)

    for _ in range(max_len - 1):
        logits = model(ids)[:, -1, :]
        if use_mask:
            # The mask needed is the one for the token about to be APPENDED, i.e.
            # at position len(ids). masks_for returns the mask AT each existing
            # position, so index -1 gives the mask for the token already emitted and
            # every token is generated twice. Pad by one column and take the last.
            cur = ids.cpu().numpy()
            padded = np.concatenate([cur, np.zeros((cur.shape[0], 1), dtype=cur.dtype)], axis=1)
            mk = torch.from_numpy(gm.masks_for(padded)[:, -1, :]).to(device)
            # A finished row has no legal token; let it emit EOS so the batch is
            # rectangular without corrupting the mask semantics for live rows.
            mk[done] = False
            mk[done, eos] = True
            logits = logits.masked_fill(~mk, float("-inf"))

        if project_at_inference:
            # RUNNING MASS BUDGET, not a fixed mass per token.
            #
            # Density is mass/volume, so a target density fixes a total mass budget
            # rho*V/1.66054 once the cell is known. Cell-first ordering makes V
            # determined before any element token, so the per-step target is the
            # remaining mass spread over the remaining element slots -- the same
            # running-budget form the synthetic task validated.
            #
            # Targeting a constant density value on the mass channel instead (the
            # first version of this code) asks for 5 u per token, lighter than
            # lithium, and drives the composition toward the lightest elements.
            B, Vsz = logits.shape
            live = mk if use_mask else torch.ones_like(logits, dtype=torch.bool)
            budget, slots = _mass_budget(ids.cpu().numpy(), V, cfg)
            budget = torch.from_numpy(budget).to(device).float()
            slots = torch.from_numpy(slots).to(device).float()
            active = (slots > 0) & torch.isfinite(budget) & ~done
            ch = torch.from_numpy(mass_channel(ids.cpu().numpy(), V)).to(device)
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
            n_projected += feas.float()
        else:
            q = torch.softmax(logits, dim=-1)

        nxt = torch.multinomial(q.clamp_min(0).nan_to_num(), 1).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
        done = done | (nxt == eos)
        if bool(done.all()):
            break

    model.train()
    seqs = [[int(t) for t in row if True] for row in ids.cpu().numpy()]
    tot = (n_infeasible + n_projected).clamp_min(1)
    return seqs, {
        "per_step_infeasible_fraction": float((n_infeasible / tot).mean()),
        "mean_steps_projected": float(n_projected.mean()),
    }


def structural_metrics(seqs, V, gm, cfg) -> dict:
    """Validity and density statistics over generated sequences."""
    dens, valid, complete = [], 0, 0
    for s in seqs:
        toks = [V.tokens[i] for i in s]
        ok = (toks.count("<COMP>") == 1 and toks.count("<END_COMP>") == 1
              and toks.count("<CELL>") == 1 and toks.count("<END_CELL>") == 1
              and sum(t.startswith("len:") for t in toks) == 3
              and sum(t.startswith("ang:") for t in toks) == 3
              and sum(t.startswith("el:") for t in toks)
              == sum(t.startswith("cnt:") for t in toks)
              and sum(t.startswith("el:") for t in toks) >= 1)
        valid += ok
        d = V.decode_density(s) if ok else None
        if d is not None and np.isfinite(d) and 0 < d < 100:
            complete += 1
            dens.append(d)
    d = np.asarray(dens)
    return {
        "n_samples": len(seqs),
        "structural_validity_rate": valid / len(seqs),
        "density_computable_rate": complete / len(seqs),
        "mean_density": float(d.mean()) if len(d) else None,
        "median_density": float(np.median(d)) if len(d) else None,
        "density_sd": float(d.std()) if len(d) else None,
        "density_error_vs_target": (float(d.mean() - cfg.density_target)
                                    if len(d) else None),
    }


def main(smoke: bool = False, arm: str = "A2", steps: int | None = None) -> Path:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CrystalTrainConfig(arm=arm, steps=steps or (300 if smoke else 6000))
    if smoke:
        cfg.eval_every, cfg.n_eval_samples, cfg.train_limit = 150, 512, 2000

    train = load_split("train", limit=cfg.train_limit)
    V = CrystalVocab(ELEMENTS, order=cfg.order)
    gm = GrammarMask(V, order=cfg.order)
    enc = [V.encode(c) for c in train]
    max_len = max(len(e) for e in enc)
    data = pad_batch(enc, V.index["<EOS>"])
    data_density = np.array([c.density for c in train])

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = CrystalDecoder(len(V), max_len + 2, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    mass_t = torch.tensor([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                           for t in V.tokens], device=dev, dtype=torch.float32)

    history, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        idx = rng.integers(0, len(data), cfg.batch)
        x = torch.from_numpy(data[idx]).to(dev)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)

        if cfg.arm != "A1":
            mk = torch.from_numpy(gm.masks_for(x.cpu().numpy())[:, 1:, :]).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))

        if cfg.arm == "A3":
            B, L, Vs = logits.shape
            flat = logits.reshape(B * L, Vs)
            mkf = mk.reshape(B * L, Vs)
            # Same running-budget target as inference, computed per prefix. Training
            # and deployment must solve the identical problem, or the layer is
            # trained against a constraint the deployed system never applies.
            bud = np.empty((B, L)); slt = np.empty((B, L))
            chan = np.empty((B, L, Vs), dtype=np.float32)
            xc = x.cpu().numpy()
            for t in range(L):
                bb_, ss_ = _mass_budget(xc[:, :t + 1], V, cfg)
                bud[:, t], slt[:, t] = bb_, ss_
                chan[:, t] = mass_channel(xc[:, :t + 1], V)
            A = torch.from_numpy(chan).to(dev).reshape(B * L, 1, Vs).contiguous()
            attain = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("nan")))
            lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
            hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values
            bt = torch.from_numpy(bud).to(dev).float().reshape(-1)
            st = torch.from_numpy(slt).to(dev).float().reshape(-1)
            act = (st > 0) & torch.isfinite(bt)
            b = torch.where(act, bt / st.clamp_min(1), lo).clamp(min=lo, max=hi).unsqueeze(-1)
            res = project(flat, A, b, mkf)
            q = torch.where(res.converged.unsqueeze(-1), res.q,
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
                                use_mask=(cfg.arm != "A1"), project_at_inference=False,
                                max_len=max_len + 2, rng=rng)
            m = structural_metrics(seqs, V, gm, cfg)
            m |= {"step": step, "loss": float(loss.item()),
                  "elapsed_s": round(time.time() - t0, 1)} | diag
            if cfg.arm == "A3":
                sd, dg = sample(model, V, gm, cfg, dev, cfg.n_eval_samples,
                                use_mask=True, project_at_inference=True,
                                max_len=max_len + 2, rng=rng)
                m["deployed"] = structural_metrics(sd, V, gm, cfg) | dg
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
        "data_density": {
            "mean": float(data_density.mean()), "median": float(np.median(data_density)),
            "sd": float(data_density.std()),
        },
        "history": history,
        "final": history[-1],
        "wall_clock_s": round(time.time() - t0, 1),
    }
    return emit(
        f"WS3_crystal_{cfg.arm}" + ("_smoke" if smoke else ""),
        payload,
        config={"arm": cfg.arm, "steps": cfg.steps, "smoke": smoke},
        seeds=[cfg.seed], script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", default="A2", choices=["A1", "A2", "A3"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    print(main(smoke=a.smoke, arm=a.arm, steps=a.steps))
