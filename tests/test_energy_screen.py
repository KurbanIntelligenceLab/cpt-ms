"""The energy surrogate must be calibrated, and labelled as a surrogate.

This is an ML-potential screen, not DFT. Two failure modes are pinned: reporting
agreement on the structures the elemental references were fitted to, and letting a
rank-deficient or rare-element-driven fit pass as calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.crystal.energy_screen import MIN_OCCURRENCES, fit_references, formation_energy

ART = Path("results/WS3_energy_calibration.json")


@pytest.fixture(scope="module")
def cal():
    if not ART.exists():
        pytest.skip("calibration artifact absent; run scripts.crystal.energy_screen")
    return json.load(open(ART))["payload"]


def test_artifact_declares_it_is_not_dft(cal):
    """A surrogate presented as DFT would misrepresent every downstream claim."""
    assert cal["is_dft"] is False
    assert "not DFT" in cal["disclaimer"]
    assert "CHGNet" in cal["surrogate"]


def test_agreement_is_reported_on_a_held_out_split(cal):
    """References fitted on one half, agreement measured on the other."""
    c = cal["calibration"]
    assert c["n_fit_split"] > 0 and c["n_test_split"] > 0
    assert c["n_fit_split"] + c["n_test_split"] == c["n_scored"]
    assert cal["held_out_agreement"]["n_evaluated"] <= c["n_test_split"]


def test_fit_is_not_rank_deficient(cal):
    """A rank-deficient reference fit is under-determined, not a calibration."""
    c = cal["calibration"]
    assert c["rank_deficient"] is False
    assert c["design_matrix_rank"] == c["n_element_references"]


def test_agreement_is_good_enough_to_screen_with(cal):
    """MAE must be small against the spread of the labels it is predicting.

    An MAE comparable to the label spread would mean the screen carries no
    information, however high its correlation.
    """
    h = cal["held_out_agreement"]
    assert h["mae_ev_per_atom"] < 0.05, f"MAE {h['mae_ev_per_atom']:.4f} eV/atom"
    assert h["mae_ev_per_atom"] < 0.1 * h["label_spread_ev_per_atom"]
    assert abs(h["bias_ev_per_atom"]) < 0.01


def test_rare_element_structures_are_excluded_and_counted(cal):
    """Exclusions must be visible, not silent."""
    h = cal["held_out_agreement"]
    assert "n_excluded_rare_elements" in h
    assert h["n_excluded_rare_elements"] >= 0
    assert MIN_OCCURRENCES >= 5


def test_formation_energy_returns_none_for_unreferenced_elements():
    """A missing reference must fail loudly rather than silently score as zero."""
    rec = {"comp": {"Xx": 1.0}, "n_atoms": 1, "e_total": -5.0}
    assert formation_energy(rec, {"Fe": -8.0}) is None
    ok = {"comp": {"Fe": 2.0}, "n_atoms": 2, "e_total": -16.0}
    assert formation_energy(ok, {"Fe": -8.0}) == pytest.approx(0.0)


def test_reference_fit_refuses_an_empty_split():
    with pytest.raises(ValueError):
        fit_references([{"comp": {"Fe": 1.0}, "n_atoms": 1, "e_total": -8.0,
                         "label_ef": 0.0}], min_occ=99)


def test_disclaimer_does_not_claim_dft_is_unavailable(cal):
    """The surrogate must be justified by what it is for, not by a false scarcity.

    An earlier version of this disclaimer asserted no DFT engine was available. That was
    never tested against GPAW, the research plan's designated engine, and it is false:
    GPAW 25.7.0 installs and runs on the compute host. The surrogate's justification is
    that MP-20 already carries DFT labels for these reference structures. Stating a
    capability gap that does not exist would misrepresent the work's constraints.
    """
    d = cal["disclaimer"]
    assert "No DFT engine is available" not in d
    assert "GPAW" in d, "the disclaimer must record that a DFT engine IS available"
    assert "not because DFT is unavailable" in d


def test_disclaimer_names_the_real_blocker_for_generated_samples(cal):
    """The blocker is missing atomic positions, which stops DFT and the surrogate alike."""
    d = cal["disclaimer"]
    assert "atomic positions" in d
    assert "GENERATED" in d
