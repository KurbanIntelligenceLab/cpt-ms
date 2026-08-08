"""Structural lemmas relating the tilted and hard-conditioned targets.

Both properties below were raised as candidate theorems rather than artifacts of
one seed. They are verified here against random base distributions and across
constraint levels, not only on the project's Markov chain.

Lemma 1 (mass-displacement identity).
    TV(tilt, hard) == mass the tilt places outside the hard-feasible set.

    Proof. On the hard-feasible set F = {x : phi(x) = c}, phi is constant, so
    exp(lambda*phi(x)) = exp(lambda*c) is constant on F. Writing Z for the
    tilt's normalizer over the S-feasible set and Z_F for the base's mass on F,
    the tilt on F is p(x)exp(lambda c)/Z while hard is p(x)/Z_F. Since
    Z >= Z_F exp(lambda c) (the tilt's normalizer also collects mass off F,
    where the integrand is non-negative), tilt(x) <= hard(x) for every x in F.
    Hard puts zero mass off F, so all of the total variation is carried by the
    tilt's off-F mass, giving TV = tilt(F^c) exactly.

    Note the argument nowhere requires c = 0: phi is constant on F for any c,
    which is all the proof uses. The tests cover c != 0 accordingly.

Lemma 2 (I-projection ordering).
    KL(tilt || base) <= KL(hard || base).

    Both distributions satisfy E[phi] = c, so both lie in the same moment-
    constrained set; the tilt is the I-projection of the base onto that set and
    is therefore the minimizer. The hard-conditioned distribution is a feasible
    but suboptimal member. The inequality is strict whenever the two differ.

    Corollary (used when reading reported numbers): the KL *gap* between the two
    is invariant to whether KL is measured against the raw base or against the
    base conditioned on a superset support containing both, since that
    conditioning shifts both KLs by the same normalizing constant.
"""

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
    """Charge levels reachable with positive mass under the S-feasible support."""
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
    """The KL gap does not depend on which nested base the KLs are measured against.

    Reported absolute KLs shift with the choice of reference; the gap does not.
    This is why two correct analyses can report different absolute values —
    e.g. against the raw data distribution versus the S-conditioned base — while
    agreeing exactly on the separation between the targets.
    """
    data = task.p_data
    s_cond = conditioning_target(task, apply_S=True)
    tilt, _ = tilting_target(task, apply_S=True, charge_target=0.0)
    hard = conditioning_target(task, apply_S=True, charge_equals=0)

    gap_data = kl(hard, data) - kl(tilt, data)
    gap_scond = kl(hard, s_cond) - kl(tilt, s_cond)
    assert gap_data == pytest.approx(gap_scond, abs=1e-9)

    # The shift itself is the log-reciprocal of the base mass on the S-feasible set.
    shift = kl(tilt, s_cond) - kl(tilt, data)
    assert shift == pytest.approx(np.log(float(data[task.feasible_S].sum())), abs=1e-9)
