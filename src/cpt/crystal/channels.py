"""Constraint channels: what each token contributes, and how much budget is left."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cpt.crystal.mp20 import ATOMIC_MASS, CrystalVocab


def weight_channel(ids: np.ndarray, V: CrystalVocab,
                   table: dict[str, float]) -> np.ndarray:
    """Per-token contribution of ANY additive per-element quantity, given the prefix."""
    n = ids.shape[0]
    out = np.zeros((n, len(V)), dtype=np.float32)
    cnt_idx = [(i, int(t[4:])) for i, t in enumerate(V.tokens) if t.startswith("cnt:")]
    el_w = np.array([table.get(t[3:], 0.0) if t.startswith("el:") else 0.0
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
            w = table.get(pending, 0.0)
            for i, k in cnt_idx:
                out[r, i] = w * k
        else:
            out[r] = el_w
    return out


MASS_UNIT = 1.66053906660
"""Divisor turning a g/cm3 density target into amu per cubic angstrom."""


def mass_channel(ids: np.ndarray, V: CrystalVocab) -> np.ndarray:
    """Per-token mass contribution GIVEN the prefix. Shape (n, vocab)."""
    return weight_channel(ids, V, ATOMIC_MASS)


def _element_position(ids: np.ndarray, V: CrystalVocab) -> np.ndarray:
    n = ids.shape[0]
    out = np.zeros(n, dtype=bool)
    for r in range(n):
        state = None
        for t in range(ids.shape[1]):
            tok = V.tokens[ids[r, t]]
            if tok == "<COMP>":
                state = "el"
            elif tok.startswith("el:"):
                state = "cnt"
            elif tok.startswith("cnt:"):
                state = "el"
            elif tok in ("<END_COMP>", "<EOS>"):
                state = None
        out[r] = state == "el"
    return out


def _additive_budget(ids: np.ndarray, V: CrystalVocab, cfg, table: dict[str, float],
                     target: float, unit_factor: float) -> tuple[np.ndarray, np.ndarray]:
    n, L = ids.shape
    budget = np.full(n, np.nan, dtype=np.float64)
    slots = np.zeros(n, dtype=np.float64)
    for r in range(n):
        lens, angs, amount, n_el, cell_closed = [], [], 0.0, 0, False
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
                amount += table.get(cur_el, 0.0) * k
                n_el += k
                cur_el = None
        if cell_closed:
            vol = V.cell_volume(lens, angs)
            if vol:
                total = target * vol / unit_factor
                budget[r] = max(total - amount, 0.0)
                slots[r] = max(cfg.expected_atoms - n_el, 1)
    return budget, slots


def _mass_budget(ids: np.ndarray, V: CrystalVocab, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Mass still owed to hit the density target, and the slots left to spend it over."""
    return _additive_budget(ids, V, cfg, ATOMIC_MASS, cfg.density_target, MASS_UNIT)


@dataclass(frozen=True)
class Constraint:
    """One channel's projection inputs at every prefix position of a flattened batch."""

    A: torch.Tensor       # (n, 1, vocab) per-token contribution
    b: torch.Tensor       # (n, 1) per-step target, clamped into the attainable range
    active: torch.Tensor  # (n,) where the constraint actually binds
    budget: torch.Tensor  # (n,) raw quantity still owed, before dividing over slots


def budget_constraint(flat, mkf, xc, V, cfg, dev, channel, budget_fn) -> Constraint:
    """Build the projection inputs for one additive channel.

    Every steered arm computes this identically and differs only in `channel` (the
    per-token contributions) and `budget_fn` (how much of the quantity is still owed).
    `channel` is either a vocabulary-sized tensor, reused at every position, or a
    callable returning the contributions for a prefix, when they depend on it.
    """
    n_flat, vocab = flat.shape
    n, L = xc.shape[0], n_flat // xc.shape[0]

    bud = np.empty((n, L))
    slt = np.empty((n, L))
    per_prefix = callable(channel)
    chan = np.empty((n, L, vocab), dtype=np.float32) if per_prefix else None
    for t in range(L):
        bud[:, t], slt[:, t] = budget_fn(xc[:, :t + 1])
        if per_prefix:
            chan[:, t] = channel(xc[:, :t + 1])

    if per_prefix:
        A = torch.from_numpy(chan).to(dev).reshape(n_flat, 1, vocab).contiguous()
    else:
        A = channel.view(1, 1, vocab).expand(n_flat, 1, vocab).contiguous()

    # Only tokens the grammar allows can carry the quantity, so the attainable range is
    # taken over those; b outside it would make the projection infeasible by construction.
    attain = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("nan")))
    lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
    hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values

    bt = torch.from_numpy(bud).to(dev).float().reshape(-1)
    st = torch.from_numpy(slt).to(dev).float().reshape(-1)
    elp = np.empty((n, L), dtype=bool)
    for t in range(L):
        elp[:, t] = _element_position(xc[:, :t + 1], V)
    ept = torch.from_numpy(elp).to(dev).reshape(-1)

    active = (st > 0) & torch.isfinite(bt) & ept
    b = torch.where(active, bt / st.clamp_min(1), lo).clamp(min=lo, max=hi).unsqueeze(-1)
    return Constraint(A=A, b=b, active=active, budget=bt)
