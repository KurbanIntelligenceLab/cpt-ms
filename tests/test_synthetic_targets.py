"""Verification for the synthetic task and its exact targets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.synthetic.task import TaskConfig, enumerate_task, prefix_allowed_mask
from scripts.synthetic.targets import (
    conditioning_target,
    kl,
    solve_tilt,
    tilting_target,
    violation_rate,
)

SMALL = TaskConfig(length=5, max_count=2)


@pytest.fixture(scope="module")
def task():
    return enumerate_task(SMALL)


def test_enumeration_is_a_distribution(task):
    assert task.sequences.shape[0] == SMALL.vocab_size ** SMALL.length
    assert np.isclose(task.p_data.sum(), 1.0, atol=1e-12)
    assert len(np.unique(task.sequences, axis=0)) == task.sequences.shape[0]


def test_data_probabilities_match_sampling(task):
    """Enumerated probabilities agree with forward sampling from the same chain."""
    rng = np.random.default_rng(12345)
    n = 400_000
    L, V = SMALL.length, SMALL.vocab_size
    draws = np.empty((n, L), dtype=np.int8)
    draws[:, 0] = rng.choice(V, size=n, p=task.initial)
    for t in range(1, L):
        for v in range(V):
            idx = np.flatnonzero(draws[:, t - 1] == v)
            if idx.size:
                draws[idx, t] = rng.choice(V, size=idx.size, p=task.transition[v])

    powers = (V ** np.arange(L - 1, -1, -1)).astype(np.int64)
    emp = np.bincount(draws.astype(np.int64) @ powers, minlength=V ** L) / n
    assert 0.5 * np.abs(emp - task.p_data).sum() < 0.02


def test_regime_S_mask_matches_brute_force(task):
    """The prefix mask admits exactly the tokens that keep a sequence feasible."""
    rng = np.random.default_rng(7)
    feasible_seqs = task.sequences[task.feasible_S]
    for _ in range(200):
        seq = feasible_seqs[rng.integers(len(feasible_seqs))]
        t = int(rng.integers(0, SMALL.length))
        prefix = tuple(int(x) for x in seq[:t])
        mask = prefix_allowed_mask(SMALL, prefix)
        for v in range(SMALL.vocab_size):
            counts = np.bincount(np.array(prefix + (v,)), minlength=SMALL.vocab_size)
            extendable = bool((counts <= SMALL.max_count).all())
            assert bool(mask[v]) == extendable


def test_tilt_satisfies_expectation_constraint(task):
    """The solved multiplier reproduces the target mean to machine precision."""
    for target in (-2.0, 0.0, 1.5):
        q, lam = tilting_target(task, apply_S=True, charge_target=target)
        assert np.isclose(q.sum(), 1.0, atol=1e-12)
        assert np.isclose(q @ task.total_charge, target, atol=1e-9)


def test_tilt_is_kl_optimal(task):
    """No feasible perturbation meeting the constraint lowers KL to the base."""
    base = conditioning_target(task, apply_S=True)
    q_star, _ = tilting_target(task, apply_S=True, charge_target=0.0)
    kl_star = kl(q_star, base)

    rng = np.random.default_rng(3)
    phi = task.total_charge.astype(np.float64)
    support = np.flatnonzero(base > 0)
    for _ in range(50):
        pert = np.zeros_like(q_star)
        pert[support] = rng.normal(0, 1, size=support.size) * q_star[support]
        A = np.vstack([np.ones_like(phi[support]), phi[support]])
        coef, *_ = np.linalg.lstsq(A.T, np.vstack([pert[support]]).T, rcond=None)
        pert[support] -= A.T @ coef.ravel()
        for scale in (0.05, 0.15):
            q = q_star + scale * pert
            if np.any(q[support] <= 0):
                continue
            q = q / q.sum()
            if not np.isclose(q @ phi, 0.0, atol=1e-8):
                continue
            assert kl(q, base) >= kl_star - 1e-12


def test_expectation_target_still_violates_per_sample(task):
    """The core scoping claim: expectation satisfaction is not per-sample."""
    q, _ = tilting_target(task, apply_S=True, charge_target=0.0)
    assert np.isclose(q @ task.total_charge, 0.0, atol=1e-9)
    assert violation_rate(task, q) > 0.5


def test_hard_and_tilt_targets_are_distinct(task):
    """Conditioning and tilting give different distributions."""
    hard = conditioning_target(task, apply_S=True, charge_equals=0)
    tilt, _ = tilting_target(task, apply_S=True, charge_target=0.0)
    assert violation_rate(task, hard) == pytest.approx(0.0, abs=1e-12)
    assert 0.5 * np.abs(hard - tilt).sum() > 0.1


def test_infeasible_target_raises(task):
    """A target outside the attainable range fails loudly, not silently."""
    with pytest.raises(ValueError):
        solve_tilt(task, apply_S=True, charge_target=1e6)
