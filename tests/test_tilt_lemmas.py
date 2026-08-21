"""Structural lemmas relating the tilted and hard-conditioned targets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.synthetic.task import TaskConfig, enumerate_task
from scripts.synthetic.targets import (
    _normalize_log,
    conditioning_target,
    kl,
    tilting_target,
    tv,
    violation_rate,
)

CONFIG = TaskConfig(length=6, max_count=2)


@pytest.fixture(scope="module")
def task():
    return enumerate_task(CONFIG)


def _attainable(task, apply_S=True):
    sel = task.feasible_S if apply_S else np.ones(len(task.total_charge), bool)
    vals, counts = np.unique(task.total_charge[sel], return_counts=True)
    return [int(v) for v, n in zip(vals, counts) if n >= 2]


@pytest.mark.parametrize("c", [-2, 0, 2])
def test_lemma1_tv_equals_violating_mass(task, c):
    """TV(tilt, hard) equals the tilt's off-feasible mass, exactly."""
    if c not in _attainable(task):
        pytest.skip(f"charge level {c} not attainable on this task")
    tilt, _ = tilting_target(task, apply_S=True, charge_target=float(c))
    hard = conditioning_target(task, apply_S=True, charge_equals=c)
    assert tv(tilt, hard) == pytest.approx(violation_rate(task, tilt, charge_equals=c), abs=1e-12)


@pytest.mark.parametrize("c", [-2, 0, 2])
def test_lemma1_tilt_dominated_on_feasible_set(task, c):
    """The mechanism behind Lemma 1: tilt <= hard pointwise on the feasible set."""
    if c not in _attainable(task):
        pytest.skip(f"charge level {c} not attainable on this task")
    tilt, _ = tilting_target(task, apply_S=True, charge_target=float(c))
    hard = conditioning_target(task, apply_S=True, charge_equals=c)
    F = task.total_charge == c
    assert np.all(tilt[F] <= hard[F] + 1e-15)


def test_lemma1_holds_on_random_bases(task):
    """Lemma 1 is a property of the construction, not of the Markov chain."""
    rng = np.random.default_rng(11)
    for _ in range(25):
        log_base = rng.normal(0, 3, size=len(task.log_p_data))
        tilt, _ = tilting_target(task, log_base=log_base, apply_S=True, charge_target=0.0)
        hard = conditioning_target(task, log_base=log_base, apply_S=True, charge_equals=0)
        assert tv(tilt, hard) == pytest.approx(violation_rate(task, tilt), abs=1e-12)


def test_lemma2_tilt_is_closer_than_hard(task):
    """KL(tilt || base) <= KL(hard || base) on random bases."""
    rng = np.random.default_rng(23)
    for _ in range(25):
        log_base = rng.normal(0, 3, size=len(task.log_p_data))
        base = _normalize_log(np.where(task.feasible_S, log_base, -np.inf))
        tilt, _ = tilting_target(task, log_base=log_base, apply_S=True, charge_target=0.0)
        hard = conditioning_target(task, log_base=log_base, apply_S=True, charge_equals=0)
        assert kl(tilt, base) <= kl(hard, base) + 1e-12


def test_lemma2_gap_is_reference_invariant(task):
    """The KL gap does not depend on which nested base the KLs are measured against."""
    data = task.p_data
    s_cond = conditioning_target(task, apply_S=True)
    tilt, _ = tilting_target(task, apply_S=True, charge_target=0.0)
    hard = conditioning_target(task, apply_S=True, charge_equals=0)

    gap_data = kl(hard, data) - kl(tilt, data)
    gap_scond = kl(hard, s_cond) - kl(tilt, s_cond)
    assert gap_data == pytest.approx(gap_scond, abs=1e-9)

    shift = kl(tilt, s_cond) - kl(tilt, data)
    assert shift == pytest.approx(np.log(float(data[task.feasible_S].sum())), abs=1e-9)
