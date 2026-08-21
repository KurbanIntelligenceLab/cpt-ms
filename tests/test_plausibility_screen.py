"""Plausibility screens must be measured against the data, not against 100%."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.crystal.plausibility_screen import _smact_screens, screen_compositions

from _devtree import requires

_needs_smact = requires("smact")

ART = Path("results/plausibility_reference.json")


@pytest.fixture(scope="module")
def ref():
    if not ART.exists():
        pytest.skip("reference absent; run scripts.crystal.plausibility_screen")
    return json.load(open(ART))["payload"]


def test_the_data_reference_is_measured_not_assumed(ref):
    """The ceiling must come from scoring real structures through the same path."""
    r = ref["data_reference"]
    assert r["n_screened"] >= 100
    assert r["charge_balanced_rate"] < 1.0, (
        "a reference of exactly 1.0 would mean the screen is not discriminating"
    )
    assert "IDENTICAL path" in ref["note"]


def test_the_energy_limitation_is_stated_in_the_artifact(ref):
    """Why an energy screen does not apply to generated samples must be on record."""
    w = ref["why_not_energy"]
    assert "no atomic positions" in w
    assert "structure prediction" in w
    assert "initialization" in w


def test_packing_window_comes_from_the_data_percentiles(ref):
    w = ref["packing_window_a3_per_atom"]
    assert 5.0 < w["lo_1pct"] < w["data_mean"] < w["hi_99pct"] < 100.0


@_needs_smact
def test_charge_screen_accepts_a_known_balanced_composition():
    """NaCl must balance; a screen that rejects it is broken."""
    balanced, ordered = _smact_screens({"Na": 1, "Cl": 1})
    assert balanced and ordered


@_needs_smact
def test_charge_screen_rejects_an_all_cation_composition():
    """Two alkali metals cannot balance to zero charge."""
    balanced, _ = _smact_screens({"Na": 1, "K": 1})
    assert not balanced


@_needs_smact
def test_packing_screen_rejects_an_impossibly_dense_cell():
    """A cell far too small for its atom count must fail regardless of chemistry."""
    out = screen_compositions([({"Na": 1, "Cl": 1}, 1.0, 2)], 8.9, 46.3)
    assert out["packing_plausible_rate"] == 0.0
    ok = screen_compositions([({"Na": 1, "Cl": 1}, 40.0, 2)], 8.9, 46.3)
    assert ok["packing_plausible_rate"] == 1.0


@_needs_smact
def test_screen_skips_degenerate_inputs_rather_than_crashing():
    out = screen_compositions([({}, 10.0, 2), ({"Na": 1}, 0.0, 1),
                               ({"Na": 1, "Cl": 1}, 40.0, 2)], 8.9, 46.3)
    assert out["n_screened"] == 1
