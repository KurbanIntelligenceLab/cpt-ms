from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.mp20 import ELEMENTS, CrystalVocab, load_split
from scripts.layer.projection import project
from scripts.train.train_crystal import (
    CrystalDecoder,
    CrystalTrainConfig,
    GrammarMask,
    _element_position,
    pad_batch,
    weight_channel,
)

SCRIPT = "scripts/crystal/surrogate_phi.py"

ELECTRONEG = {
    "H": 2.2, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.9, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.0, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
    "Co": 1.88, "Ni": 1.91, "Cu": 1.9, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55,
    "Br": 2.96, "Kr": 3.0, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16,
    "Tc": 1.9, "Ru": 2.2, "Rh": 2.28, "Pd": 2.2, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.1, "I": 2.66, "Xe": 2.6, "Cs": 0.79, "Ba": 0.89, "La": 1.1, "Ce": 1.12,
    "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.2, "Gd": 1.2, "Tb": 1.1, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.1, "Lu": 1.27, "Hf": 1.3, "Ta": 1.5, "W": 2.36,
    "Re": 1.9, "Os": 2.2, "Ir": 2.2, "Pt": 2.28, "Au": 2.54, "Hg": 2.0, "Tl": 1.62, "Pb": 2.33,
    "Bi": 2.02, "Po": 2.0, "At": 2.2, "Rn": 2.2, "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3,
    "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28,
}


def measure_nonadditivity(n_structures: int) -> dict:
    train = load_split("train", limit=n_structures)
    usable = [c for c in train if all(e in ELECTRONEG for e in c.formula)]
    el_index = {e: i for i, e in enumerate(ELEMENTS)}
    X = np.zeros((len(usable), len(ELEMENTS)), dtype=np.float64)
    y = np.zeros(len(usable), dtype=np.float64)
    for i, c in enumerate(usable):
        n_atoms = sum(c.formula.values())
        s = sum(ELECTRONEG[e] * n for e, n in c.formula.items())
        y[i] = s / n_atoms
        for e, n in c.formula.items():
            X[i, el_index[e]] = n
    Xd = np.hstack([X, np.ones((len(usable), 1))])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ coef
    ssr = float(np.sum((yhat - y) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(ssr / len(y)))
    return {
        "n_structures_total": len(train),
        "n_structures_usable": len(usable),
        "excluded_missing_electronegativity": len(train) - len(usable),
        "target_mean": float(y.mean()),
        "target_sd": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "linear_in_token_counts_r2": float(1.0 - ssr / sst),
        "linear_in_token_counts_rmse": rmse,
        "linear_in_token_counts_relative_rmse": float(rmse / y.std()),
        "verdict": (
            "mean electronegativity is a ratio of two additive quantities "
            "(sum of per-element electronegativity over atom count), so the best "
            "fixed-per-element-weight linear model in token counts leaves substantial "
            "unexplained variance: this is the measured degree of non-additivity"
        ),
    }


def build_atoms_predictor_dataset(structures) -> tuple[np.ndarray, np.ndarray]:
    rows_X, rows_y = [], []
    for c in structures:
        if not all(e in ELECTRONEG for e in c.formula):
            continue
        items = sorted(c.formula.items())
        n_tot = sum(k for _, k in items)
        cum_n = 0
        for e, k in items:
            rows_X.append((c.volume, cum_n))
            rows_y.append(n_tot - cum_n)
            cum_n += k
    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.float32)


class RemainingAtomsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_remaining_atoms_predictor(structures, seed: int, n_epochs: int = 300) -> dict:
    X, y = build_atoms_predictor_dataset(structures)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xn = (X - mu) / sd

    const_pred = np.full(len(te), y[tr].mean())
    rmse_const = float(np.sqrt(np.mean((const_pred - y[te]) ** 2)))

    torch.manual_seed(seed)
    net = RemainingAtomsMLP()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(Xn[tr])
    ytr_t = torch.tensor(y[tr])
    Xte_t = torch.tensor(Xn[te])
    yte_t = torch.tensor(y[te])
    for _ in range(n_epochs):
        opt.zero_grad()
        pred = net(Xtr_t)
        loss = ((pred - ytr_t) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mlp_pred = net(Xte_t).numpy()
    rmse_mlp = float(np.sqrt(np.mean((mlp_pred - y[te]) ** 2)))

    stats = {
        "n_rows": len(X), "n_train": len(tr), "n_test": len(te),
        "target_mean": float(y.mean()), "target_sd": float(y.std()),
        "held_out_rmse_constant_baseline": rmse_const,
        "held_out_rmse_learned_mlp": rmse_mlp,
        "improvement_fraction": float(1.0 - rmse_mlp / rmse_const),
        "feature_mean": mu.tolist(), "feature_sd": sd.tolist(),
    }
    return net, mu, sd, stats


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
            pred = net(torch.tensor(xn)).numpy()
        return pred
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
        ok = (toks.count("<COMP>") == 1 and toks.count("<END_COMP>") == 1
              and toks.count("<CELL>") == 1 and toks.count("<END_CELL>") == 1
              and sum(t.startswith("len:") for t in toks) == 3
              and sum(t.startswith("ang:") for t in toks) == 3
              and sum(t.startswith("el:") for t in toks)
              == sum(t.startswith("cnt:") for t in toks)
              and sum(t.startswith("el:") for t in toks) >= 1)
        valid += ok
        if ok:
            f = {}
            els = [t[3:] for t in toks if t.startswith("el:")]
            cnts = [int(t[4:]) for t in toks if t.startswith("cnt:")]
            for e, k in zip(els, cnts):
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
    history = []
    for step in range(1, cfg.steps + 1):
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
            bud = np.empty((B, L)); slt = np.empty((B, L))
            for t in range(L):
                bb_, ss_ = _electroneg_budget(xc[:, :t + 1], V, cfg.electroneg_target, slots_fn)
                bud[:, t], slt[:, t] = bb_, ss_
            A = en_t.view(1, 1, Vs).expand(B * L, 1, Vs).contiguous()
            attain = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("nan")))
            lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
            hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values
            bt = torch.from_numpy(bud).to(dev).float().reshape(-1)
            st = torch.from_numpy(slt).to(dev).float().reshape(-1)
            elp = np.empty((B, L), dtype=bool)
            for t in range(L):
                elp[:, t] = _element_position(xc[:, :t + 1], V)
            ept = torch.from_numpy(elp).to(dev).reshape(-1)
            act = (st > 0) & torch.isfinite(bt) & ept
            b = torch.where(act, bt / st.clamp_min(1), lo).clamp(min=lo, max=hi).unsqueeze(-1)
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
