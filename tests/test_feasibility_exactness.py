"""Feasibility must be EXACT, and the evaluation must not manufacture a leak.

The strongest claim in the project is that a masked model places identically zero
mass outside the feasible set -- not "small", zero, because masking makes infeasible
sequences unreachable. A scoring path that floors probabilities at a positive
constant silently converts that exact zero into a tiny positive number, which is
worse than a wrong number: it looks like a property of the method.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.layer.projection import project
from scripts.synthetic.task import TaskConfig, enumerate_task


@pytest.fixture(scope="module")
def small():
    cfg = TaskConfig(length=5, max_count=2)
    return cfg, enumerate_task(cfg)


def _masks(task, cfg):
    s = task.sequences
    n, L = s.shape
    V = cfg.vocab_size
    out = np.ones((n, L, V), dtype=bool)
    for i in range(n):
        cnt = np.zeros(V, dtype=int)
        for t in range(L):
            out[i, t] = cnt < cfg.max_count
            cnt[s[i, t]] += 1
    return torch.from_numpy(out)


def _score(task, cfg, clamp: float | None):
    """Score every sequence through the projected path, with or without a floor."""
    torch.manual_seed(0)
    V, L = cfg.vocab_size, cfg.length
    seqs = torch.from_numpy(task.sequences.astype(np.int64))
    N = seqs.shape[0]
    mk = _masks(task, cfg)
    logits = torch.randn(N, L, V).masked_fill(~mk, float("-inf"))
    phi = torch.tensor(np.asarray(cfg.charges, dtype=float), dtype=torch.float32)
    flat = logits.reshape(N * L, V)
    A = phi.view(1, 1, V).expand(N * L, 1, V).contiguous()
    acc = torch.cat([torch.zeros(N, 1), phi[seqs].cumsum(-1)[:, :-1]], dim=1)
    rem = torch.arange(L, 0, -1, dtype=torch.float32).view(1, L)
    b = ((0.0 - acc) / rem).reshape(N * L, 1)
    q = project(flat, A, b, mk.reshape(N * L, V)).q
    lp = torch.log(q if clamp is None else q.clamp_min(clamp))
    tot = lp.reshape(N, L, V).gather(-1, seqs.unsqueeze(-1)).squeeze(-1).sum(-1)
    p = tot.double().exp()
    return (p / p.sum()).numpy()


def test_projection_returns_exact_zero_at_masked_entries(small):
    """The layer itself never leaks: masked entries are 0.0, not merely small."""
    cfg, task = small
    torch.manual_seed(0)
    V, L = cfg.vocab_size, cfg.length
    seqs = torch.from_numpy(task.sequences.astype(np.int64))
    N = seqs.shape[0]
    mk = _masks(task, cfg)
    logits = torch.randn(N, L, V).masked_fill(~mk, float("-inf"))
    phi = torch.tensor(np.asarray(cfg.charges, dtype=float), dtype=torch.float32)
    flat = logits.reshape(N * L, V)
    A = phi.view(1, 1, V).expand(N * L, 1, V).contiguous()
    acc = torch.cat([torch.zeros(N, 1), phi[seqs].cumsum(-1)[:, :-1]], dim=1)
    rem = torch.arange(L, 0, -1, dtype=torch.float32).view(1, L)
    b = ((0.0 - acc) / rem).reshape(N * L, 1)
    q = project(flat, A, b, mk.reshape(N * L, V)).q.reshape(N, L, V)
    assert float(q[~mk].max()) == 0.0, "projection leaked probability onto a masked token"


def test_unclamped_scoring_gives_identically_zero_infeasible_mass(small):
    """The deployed reading must reproduce the exactness claim, not approximate it."""
    cfg, task = small
    p = _score(task, cfg, clamp=None)
    assert p[~task.feasible_S].sum() == 0.0, (
        f"infeasible mass {p[~task.feasible_S].sum():.3e} is not exactly zero"
    )


def test_a_clamped_scorer_manufactures_a_leak_at_the_clamp_scale(small):
    """Pins the diagnosis: the residual came from the floor, not from the layer.

    This is the defect that was reported as float underflow. It is not underflow --
    1e-30 is representable in float64, whose denormals reach ~5e-324 -- it is a
    scoring floor. A sequence with exactly one illegal token carries the floor times
    an O(1) product of legal-step probabilities, so the leak lands within an order of
    magnitude of the clamp value itself.
    """
    cfg, task = small
    clamp = 1e-30
    leaked = _score(task, cfg, clamp=clamp)[~task.feasible_S].sum()
    assert leaked > 0.0, "clamped scorer should leak; if not, this test is not binding"
    assert clamp <= leaked <= clamp * 1e3, (
        f"leak {leaked:.3e} is not at the clamp scale {clamp:.0e}; the diagnosis "
        "that a single illegal token carries the floor would be wrong"
    )


def test_the_clamp_value_is_representable_not_underflow():
    """A representable value cannot be underflow, whatever an artifact says."""
    assert 1e-30 > np.nextafter(0.0, 1.0)
    assert 1e-30 > float(np.finfo(np.float64).tiny) / 1e100
    assert np.float64(1e-30) != 0.0


def test_regime_S_exactness_is_measured_not_hardcoded():
    """The exactness claim must read a field the artifact actually carries.

    This was previously reported from a literal 0.0 printed by an audit script, with a
    verification check written as `D['induced'][key] if key in D['induced'] else True`
    — which passed vacuously because the key did not exist. A check that cannot fail
    measures nothing, so the field is now emitted and read.
    """
    import json
    from pathlib import Path
    art = Path("results/induced_vs_tilt.json")
    if not art.exists():
        pytest.skip("artifact absent; run scripts.synthetic.induced_distribution")
    dists = json.load(open(art))["payload"]["distributions"]
    checked = 0
    for name, d in dists.items():
        if not isinstance(d, dict):
            continue
        assert "mass_outside_S" in d, (
            f"{name} has no mass_outside_S; the exactness claim would have nothing to "
            "check and any verification of it would pass vacuously"
        )
        assert d["mass_outside_S"] == 0.0, f"{name} leaks {d['mass_outside_S']:.3e}"
        checked += 1
    assert checked >= 3, f"only {checked} distributions checked; expected base_S/tilt/induced"
