"""Rejection and steering return different objects, and the baseline must be fair.

The cost argument for the projection weakens wherever rejection happens to be cheap.
The fidelity argument does not: at any target, rejection returns whatever chemistry
the data happens to contain there, and near the edges of support it returns almost
nothing. These tests pin that the comparison is measured rather than asserted, and
that the baseline arm is the generous one.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.crystal.fidelity_baseline import _composition_profile, _tv, rejection_arm
from scripts.crystal.mp20 import load_split

from _devtree import requires_mp20

pytestmark = requires_mp20


@pytest.fixture(scope="module")
def train():
    return load_split("train")


def test_tv_is_a_metric_on_the_element_distributions():
    p = {"O": 0.5, "Fe": 0.5}
    q = {"O": 0.5, "Fe": 0.5}
    assert _tv(p, q) == 0.0
    assert abs(_tv({"O": 1.0}, {"Fe": 1.0}) - 1.0) < 1e-12
    assert _tv({"O": 0.6, "Fe": 0.4}, {"O": 0.4, "Fe": 0.6}) > 0


def test_rejection_composition_departs_from_the_data_toward_the_tails(train):
    """The accepted set is a biased slice, and the bias grows toward the edges."""
    full = _composition_profile(train)["element_frequencies"]
    tvs = {}
    for t in (5.0, 10.0, 16.0, 20.0):
        arm = rejection_arm(train, t, 0.5)
        assert arm["reachable"], f"target {t} unexpectedly empty"
        tvs[t] = _tv(arm["composition"]["element_frequencies"], full)
    assert tvs[5.0] < tvs[10.0] < tvs[16.0] < tvs[20.0], (
        f"divergence should grow toward the tail: {tvs}"
    )
    assert tvs[20.0] > 0.9, "the extreme target should be nearly disjoint from the data"


def test_the_extreme_target_has_an_almost_empty_acceptance_set(train):
    """Not merely expensive: there is essentially nothing to accept."""
    arm = rejection_arm(train, 20.0, 0.5)
    assert arm["n_accepted"] < 15, f"{arm['n_accepted']} accepted; expected a handful"
    assert arm["expected_draws"] > 2000, "should be in the impractical regime"
    assert arm["composition"]["distinct_elements"] < 15, (
        "an almost-empty acceptance set cannot span the periodic table"
    )


def test_bulk_targets_are_cheap_but_already_biased(train):
    """The fidelity argument holds where the cost argument does not."""
    full = _composition_profile(train)["element_frequencies"]
    arm = rejection_arm(train, 5.0, 0.5)
    assert arm["expected_draws"] < 10, "target 5.0 should be cheap for rejection"
    assert _tv(arm["composition"]["element_frequencies"], full) > 0.15, (
        "even a cheap rejection target returns a compositionally biased sample"
    )


def test_baseline_draws_from_the_data_not_from_a_model(train):
    """The arm must be the generous one, or the comparison is rigged."""
    arm = rejection_arm(train, 5.0, 0.5)
    dens = np.array([c.density for c in train])
    direct = int((np.abs(dens - 5.0) <= 0.5).sum())
    assert arm["n_accepted"] == direct, (
        "the rejection arm must be the empirical data itself, i.e. a generator that "
        "reproduces the data distribution perfectly"
    )
