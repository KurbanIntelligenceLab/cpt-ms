"""Actionability is not determinability, and CP0 depends on the difference.

A constraint can be computable from the prefix and simultaneously frozen — no
remaining token can change its value — in which case a per-step projection has
nothing to act on there. Calibrating an active-multiplier threshold against the
determinable fraction would fail a correctly functioning layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crystal.actionability import _charge, _final_state, profile
from scripts.crystal.determinability import (
    build_b1,
    build_b1_reference,
    build_b2,
)
from scripts.crystal.family_ii import family_i_profile


def test_charge_is_determined_but_frozen_after_composition_closes():
    """The core distinction, asserted directly.

    Under the reference ordering the composition completes early, so charge is
    determined over most of the sequence — but every position after the data_ tag
    is frozen for that statistic. Determinable and actionable must therefore be
    reported separately.
    """
    toks = build_b1_reference(3, 8)
    det = family_i_profile(toks, "B1ref").usable_fraction
    act = profile(toks, "B1ref", "i_charge")
    assert det > 0.9
    assert act.actionable_fraction < 0.10
    assert det / max(act.actionable_fraction, 1e-9) > 5, (
        "determinable and actionable must differ materially here"
    )
    assert act.last_actionable is not None and act.last_actionable < 30, (
        "the actionable window must close where the composition does"
    )


def test_naive_layout_composition_parses_real_elements():
    """Regression: the fixture once emitted placeholder site labels.

    `E0`/`E1` labels parse to an empty composition, so the charge statistic
    evaluated to a meaningless zero and any measurement of family (i) on that
    fixture described the fixture rather than the format.
    """
    st = _final_state(build_b1(3, 8), "B1")
    assert st.element_counts, "composition parsed empty from the atom-site loop"
    assert sum(st.element_counts.values()) == 8, "one atom per site line"
    assert _charge(st) is not None


@pytest.mark.parametrize("backbone,builder", [
    ("B1", build_b1), ("B1ref", build_b1_reference), ("B2", build_b2),
])
@pytest.mark.parametrize("family", ["i_charge", "ii_density"])
def test_actionable_positions_are_never_more_than_the_sequence(backbone, builder, family):
    toks = builder(3, 8)
    a = profile(toks, backbone, family)
    assert 0 <= a.actionable_fraction <= 1
    assert a.n_actionable <= a.n_tokens
    if a.n_actionable:
        assert a.first_actionable <= a.last_actionable < a.n_tokens


def test_actionable_positions_actually_change_the_statistic():
    """Spot-check the counterfactual rather than trusting the counter."""
    from scripts.crystal.actionability import _candidates

    toks = build_b2(3, 8)
    a = profile(toks, "B2", "i_charge")
    assert a.n_actionable > 0
    base = _charge(_final_state(toks, "B2"))
    t = a.first_actionable
    changed = False
    for cand in _candidates(toks, t, "B2"):
        alt = list(toks)
        alt[t] = cand
        if _charge(_final_state(alt, "B2")) != base:
            changed = True
            break
    assert changed, f"position {t} reported actionable but no substitution moved charge"


def test_frozen_constraint_reports_zero_actionable():
    """A stream with no substitutable content must report frozen, not crash."""
    a = profile(["<COMP>", "<END_COMP>", "<END_LATTICE>"], "B2", "i_charge")
    assert a.actionable_fraction == 0.0
    assert a.first_actionable is None
    assert "frozen" in a.reason
