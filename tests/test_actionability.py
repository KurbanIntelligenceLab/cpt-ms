"""Actionability is not determinability, and depends on the difference."""

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
    """The core distinction, asserted directly."""
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
    """Regression: the fixture once emitted placeholder site labels."""
    st = _final_state(build_b1(3, 8), "B1")
    assert st.element_counts, "composition parsed empty from the atom-site loop"
    assert sum(st.element_counts.values()) == 8, "one atom per site line"
    assert _charge(st) is not None


@pytest.mark.parametrize("builder,backbone", [
    (build_b1, "B1"), (build_b1_reference, "B1ref"), (build_b2, "B2"),
])
def test_fixtures_use_real_elements_not_placeholders(builder, backbone):
    """Placeholder element symbols silently zero out both statistics."""
    from scripts.crystal.actionability import CHARGES, MASSES, _density

    st = _final_state(builder(3, 8), backbone)
    assert st.element_counts, "composition parsed empty"
    unknown = [el for el in st.element_counts if el not in CHARGES or el not in MASSES]
    assert not unknown, f"fixture emits elements with no charge/mass: {unknown}"
    assert _charge(st) is not None
    d = _density(st)
    assert d is not None and d > 0.0, "density must be non-zero for a real composition"


def test_density_has_more_actionable_room_than_charge_on_every_backbone():
    """Density also depends on lattice tokens, so it must have strictly more room than charge."""
    for builder, backbone in ((build_b1, "B1"), (build_b1_reference, "B1ref"),
                              (build_b2, "B2")):
        toks = builder(3, 8)
        chg = profile(toks, backbone, "i_charge").actionable_fraction
        den = profile(toks, backbone, "ii_density").actionable_fraction
        assert den > chg, f"{backbone}: density {den:.4f} not above charge {chg:.4f}"


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
