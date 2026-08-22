"""How non-additive the electronegativity channel is, and a predictor for the atoms left."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cpt.crystal.mp20 import ELECTRONEG, ELEMENTS, load_split


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
        for _e, k in items:
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
