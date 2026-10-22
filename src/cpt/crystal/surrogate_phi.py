from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cpt.common.artifacts import emit
from cpt.crystal.channels import budget_constraint
from cpt.crystal.metrics import is_structurally_valid
from cpt.crystal.mp20 import ELECTRONEG, ELEMENTS, CrystalVocab, load_split
from cpt.crystal.remaining_atoms import (
    measure_nonadditivity,
    train_remaining_atoms_predictor,
)
from cpt.layer.projection import project
from cpt.train.train_crystal import (
    CrystalDecoder,
    GrammarMask,
    _element_position,
    pad_batch,
)

SCRIPT = "src/cpt/crystal/surrogate_phi.py"


def _electroneg_budget(ids: np.ndarray, V: CrystalVocab, target: float,
                        slots_fn) -> tuple[np.ndarray, np.ndarray]:
    n, L = ids.shape
    budget = np.full(n, np.nan, dtype=np.float64)
    slots = np.zeros(n, dtype=np.float64)
    vols = np.full(n, np.nan, dtype=np.float64)
    n_committed = np.zeros(n, dtype=np.float64)
    s_committed = np.zeros(n, dtype=np.float64)
    for r in range(n):
        lens, angs, s_sum, n_el, cell_closed = [], [], 0.0, 0, False
        cur_el = None
        for t in range(L):
            tok = V.tokens[ids[r, t]]
            if tok.startswith("len:"):
                j = int(tok[4:])
                lens.append(V.LEN_LO + (j + 0.5) * (V.LEN_HI - V.LEN_LO) / V.N_LEN)
            elif tok.startswith("ang:"):
                j = int(tok[4:])
                angs.append(V.ANG_LO + (j + 0.5) * (V.ANG_HI - V.ANG_LO) / V.N_ANG)
            elif tok == "<END_CELL>":
                cell_closed = len(lens) == 3 and len(angs) == 3
            elif tok.startswith("el:"):
                cur_el = tok[3:]
            elif tok.startswith("cnt:") and cur_el:
                k = int(tok[4:])
                s_sum += ELECTRONEG.get(cur_el, 0.0) * k
                n_el += k
                cur_el = None
        if cell_closed:
            vol = V.cell_volume(lens, angs)
            if vol:
                vols[r] = vol
                n_committed[r] = n_el
                s_committed[r] = s_sum
    live = np.isfinite(vols)
    if live.any():
        remaining = slots_fn(vols[live], n_committed[live])
        remaining = np.clip(remaining, 1.0, None)
        n_hat = n_committed[live] + remaining
        s_hat = target * n_hat
        budget[live] = (s_hat - s_committed[live])
        slots[live] = remaining
    return budget, slots


def _fixed_slots(cfg_expected_atoms):
    def f(vol, n_so_far):
        return np.full_like(vol, cfg_expected_atoms) - n_so_far
    return f


def _learned_slots(net, mu, sd):
    def f(vol, n_so_far):
        x = np.stack([vol, n_so_far], axis=1).astype(np.float32)
        xn = (x - mu) / sd
        with torch.no_grad():
            return net(torch.tensor(xn)).numpy()
    return f


@dataclass
class SurrogateConfig:
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    steps: int = 600
    batch: int = 128
    lr: float = 3e-4
    warmup: int = 100
    seed: int = 0
    n_eval_samples: int = 512
    train_limit: int = 3000
    electroneg_target: float = 1.9
    expected_atoms: float = 10.4
    order: str = "cell_first"


def structural_metrics_electroneg(seqs, V, cfg, train_formulas=None, full_element_freq=None) -> dict:
    xd, valid, complete = [], 0, 0
    el_counter: Counter = Counter()
    formulas: list[tuple] = []
    for s in seqs:
        toks = [V.tokens[i] for i in s]
        ok = is_structurally_valid(toks)
        valid += ok
        if ok:
            f = {}
            els = [t[3:] for t in toks if t.startswith("el:")]
            cnts = [int(t[4:]) for t in toks if t.startswith("cnt:")]
            for e, k in zip(els, cnts, strict=False):
                f[e] = f.get(e, 0) + k
                el_counter[e] += k
            formulas.append(tuple(sorted(f.items())))
            if all(e in ELECTRONEG for e in f):
                n_tot = sum(f.values())
                s_tot = sum(ELECTRONEG[e] * k for e, k in f.items())
                complete += 1
                xd.append(s_tot / n_tot)
    x = np.asarray(xd)
    tot = sum(el_counter.values())
    freq = {e: k / tot for e, k in el_counter.items()} if tot else {}
    ent = float(-sum(p * np.log(p) for p in freq.values() if p > 0)) if freq else None
    tv = None
    if full_element_freq and freq:
        keys = set(freq) | set(full_element_freq)
        tv = 0.5 * sum(abs(freq.get(k, 0.0) - full_element_freq.get(k, 0.0)) for k in keys)
    uniq = len(set(formulas)) / len(formulas) if formulas else None
    novel = (sum(f not in train_formulas for f in formulas) / len(formulas)
             if train_formulas is not None and formulas else None)
    return {
        "n_samples": len(seqs),
        "structural_validity_rate": valid / len(seqs),
        "electroneg_computable_rate": complete / len(seqs),
        "mean_electroneg": float(x.mean()) if len(x) else None,
        "electroneg_sd": float(x.std()) if len(x) else None,
        "distinct_elements": len(freq),
        "element_entropy_nats": ent,
        "element_tv_vs_full_data": tv,
        "unique_formula_fraction": uniq,
        "novel_formula_fraction": novel,
    }


@torch.no_grad()
def sample_electroneg(model, V, gm, cfg, device, n, *, mode: str, slots_fn=None,
                      max_len: int, rng) -> tuple[list[list[int]], dict]:
    model.eval()
    ids = torch.full((n, 1), V.index["<BOS>"], dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    n_infeasible = torch.zeros(n, device=device)
    n_projected = torch.zeros(n, device=device)
    eos = V.index["<EOS>"]
    en_t = torch.tensor([ELECTRONEG.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                         for t in V.tokens], device=device, dtype=torch.float32)

    for _ in range(max_len - 1):
        logits = model(ids)[:, -1, :]
        cur = ids.cpu().numpy()
        padded = np.concatenate([cur, np.zeros((cur.shape[0], 1), dtype=cur.dtype)], axis=1)
        mk = torch.from_numpy(gm.masks_for(padded)[:, -1, :]).to(device)
        mk[done] = False
        mk[done, eos] = True
        logits = logits.masked_fill(~mk, float("-inf"))

        if mode != "unsteered":
            B, Vsz = logits.shape
            budget, slots = _electroneg_budget(ids.cpu().numpy(), V, cfg.electroneg_target, slots_fn)
            budget_t = torch.from_numpy(budget).to(device).float()
            slots_t = torch.from_numpy(slots).to(device).float()
            el_next = torch.from_numpy(_element_position(ids.cpu().numpy(), V)).to(device)
            active = (slots_t > 0) & torch.isfinite(budget_t) & ~done & el_next
            ch = en_t.unsqueeze(0).expand(B, Vsz)
            A = ch.unsqueeze(1).contiguous()
            attain = torch.where(mk, A[:, 0, :], torch.full_like(logits, float("nan")))
            lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
            hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values
            per_step = torch.where(active, budget_t / slots_t.clamp_min(1), lo)
            feas = active & (per_step >= lo - 1e-6) & (per_step <= hi + 1e-6)
            b = per_step.clamp(min=lo, max=hi).unsqueeze(-1)
            res = project(logits, A, b, mk)
            q = torch.where(res.converged.unsqueeze(-1) & feas.unsqueeze(-1),
                            res.q, torch.softmax(logits, -1))
            n_infeasible += (active & ~feas).float()
            n_projected += feas.float()
        else:
            q = torch.softmax(logits, dim=-1)

        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        bad = q.sum(-1) <= 0
        if bool(bad.any()):
            safe = torch.softmax(logits, dim=-1)
            safe = torch.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
            still_bad = bad & (safe.sum(-1) <= 0)
            q = torch.where(bad.unsqueeze(-1), safe, q)
            if bool(still_bad.any()):
                onehot = torch.zeros_like(q)
                onehot[:, eos] = 1.0
                q = torch.where(still_bad.unsqueeze(-1), onehot, q)
        nxt = torch.multinomial(q, 1).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
        done = done | (nxt == eos)
        if bool(done.all()):
            break

    model.train()
    seqs = [[int(t) for t in row] for row in ids.cpu().numpy()]
    tot = (n_infeasible + n_projected).clamp_min(1)
    return seqs, {
        "per_step_infeasible_fraction": float((n_infeasible / tot).mean()),
        "mean_steps_projected": float(n_projected.mean()),
    }


def run_arm(mode: str, cfg: SurrogateConfig, train, V, gm, max_len, data, dev,
           slots_fn, train_formulas, full_element_freq, rng, torch_seed) -> dict:
    torch.manual_seed(torch_seed)
    model = CrystalDecoder(len(V), max_len + 2, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    en_t = torch.tensor([ELECTRONEG.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                         for t in V.tokens], device=dev, dtype=torch.float32)

    t0 = time.time()
    for _step in range(1, cfg.steps + 1):
        idx = rng.integers(0, len(data), cfg.batch)
        x = torch.from_numpy(data[idx]).to(dev)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        mk = torch.from_numpy(gm.masks_for(x.cpu().numpy())[:, 1:, :]).to(dev)
        logits = logits.masked_fill(~mk, float("-inf"))

        if mode != "unsteered":
            B, L, Vs = logits.shape
            flat = logits.reshape(B * L, Vs)
            mkf = mk.reshape(B * L, Vs)
            xc = x.cpu().numpy()
            en = budget_constraint(
                flat, mkf, xc, V, cfg, dev, channel=en_t,
                budget_fn=lambda p: _electroneg_budget(p, V, cfg.electroneg_target, slots_fn))
            A, b, act = en.A, en.b, en.active
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

    seqs, diag = sample_electroneg(model, V, gm, cfg, dev, cfg.n_eval_samples,
                                   mode=mode, slots_fn=slots_fn, max_len=max_len + 2, rng=rng)
    m = structural_metrics_electroneg(seqs, V, cfg, train_formulas=train_formulas,
                                      full_element_freq=full_element_freq)
    m |= diag
    m["wall_clock_s"] = round(time.time() - t0, 1)
    m["target"] = cfg.electroneg_target
    m["error_vs_target"] = (float(m["mean_electroneg"] - cfg.electroneg_target)
                            if m["mean_electroneg"] is not None else None)
    return m


def main(smoke: bool = False) -> Path:
    cfg = SurrogateConfig(
        steps=40 if smoke else 600,
        train_limit=200 if smoke else 3000,
        n_eval_samples=64 if smoke else 512,
    )
    dev = torch.device("cpu")

    nonadd = measure_nonadditivity(400 if smoke else 8000)

    train = load_split("train", limit=cfg.train_limit)
    train = [c for c in train if all(e in ELECTRONEG for e in c.formula)]
    net, mu, sd, phi_stats = train_remaining_atoms_predictor(train, seed=cfg.seed,
                                                             n_epochs=50 if smoke else 300)

    V = CrystalVocab(ELEMENTS, order=cfg.order)
    gm = GrammarMask(V, order=cfg.order)
    enc = [V.encode(c) for c in train]
    max_len = max(len(e) for e in enc)
    data = pad_batch(enc, V.index["<EOS>"])

    full_el: Counter = Counter()
    for c in train:
        for e, k in c.formula.items():
            full_el[e] += k
    tot_el = sum(full_el.values())
    full_element_freq = {e: k / tot_el for e, k in full_el.items()}
    train_formulas = {tuple(sorted(c.formula.items())) for c in train}

    fixed_slots_fn = _fixed_slots(cfg.expected_atoms)
    learned_slots_fn = _learned_slots(net, mu, sd)

    results = {}
    for mode, slots_fn, ts in (("unsteered", None, cfg.seed),
                               ("fixed_slots", fixed_slots_fn, cfg.seed + 1),
                               ("learned_phi", learned_slots_fn, cfg.seed + 2)):
        rng = np.random.default_rng(cfg.seed if mode == "unsteered" else
                                     (cfg.seed + 1 if mode == "fixed_slots" else cfg.seed + 2))
        print(f"running arm {mode}", flush=True)
        results[mode] = run_arm(mode, cfg, train, V, gm, max_len, data, dev,
                                slots_fn, train_formulas, full_element_freq, rng, ts)
        print(f"  {mode}: {results[mode]}", flush=True)

    payload = {
        "property": "mean_electronegativity",
        "property_definition": (
            "sum over the formula of each element's Pauling electronegativity times "
            "its count, divided by the total atom count -- a ratio of two additive "
            "quantities, computable from composition alone"
        ),
        "nonadditivity_evidence": nonadd,
        "remaining_atoms_predictor": phi_stats,
        "config": asdict(cfg),
        "arms": results,
        "comparison": {
            "fixed_vs_learned_error_ratio": (
                abs(results["fixed_slots"]["error_vs_target"])
                / abs(results["learned_phi"]["error_vs_target"])
                if results["fixed_slots"]["error_vs_target"] not in (None, 0)
                and results["learned_phi"]["error_vs_target"] is not None
                else None
            ),
            "learned_vs_unsteered_error_ratio": (
                abs(results["unsteered"]["error_vs_target"])
                / abs(results["learned_phi"]["error_vs_target"])
                if results["unsteered"]["error_vs_target"] not in (None, 0)
                and results["learned_phi"]["error_vs_target"] is not None
                else None
            ),
        },
    }
    return emit("surrogate_phi" + ("_smoke" if smoke else ""), payload,
               config={"smoke": smoke}, seeds=[cfg.seed], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(smoke=a.smoke))
