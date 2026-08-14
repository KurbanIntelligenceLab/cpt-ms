"""Tests for the A7 two-constraint grid and its analysis.

The WS-5 conclusion is a NEGATIVE result, which makes it exactly the kind of claim that
rots quietly: if a later change makes the second dial steer, the report must stop saying it
does not. These tests pin the criteria the verdict is derived from, so the verdict cannot
survive a change in the underlying measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"
ART = RESULTS / "WS5_two_constraint_grid.json"
NOTES = Path(__file__).resolve().parents[1] / "reports" / "WS5_two_constraint_grid_notes.md"


def _payload():
    if not ART.exists():
        pytest.skip("WS5 grid artifact absent; run scripts.crystal.analyze_a7_grid")
    return json.loads(ART.read_text())["payload"]


def test_verdict_follows_from_the_measurements():
    """The verdict text must match what pass-through and monotonicity actually show.

    A verdict asserted beside numbers rather than derived from them is how a report comes
    to disagree with its own artifact.
    """
    p = _payload()
    s = p["summary"]
    complete = [r for r in p["rows"] if r.get("complete")]
    if not complete:
        pytest.skip("no complete rows yet")
    pt = [r["pass_through"] for r in complete if r["pass_through"] is not None]
    mono = [r["monotone_in_target"] for r in complete]
    if max(pt) < 0.25 and not any(mono):
        assert "does not steer" in s["verdict"], s["verdict"]
    elif max(pt) > 0.5 and all(mono):
        assert "steers" in s["verdict"] and "does not" not in s["verdict"], s["verdict"]
    else:
        assert "Mixed" in s["verdict"], s["verdict"]


def test_density_comparison_uses_matched_targets_only():
    """Averaging arms over different target sets would manufacture an advantage.

    The d=8.0 control sits above the reachable ceiling and carries a large error, so an
    unmatched mean would credit A7 with the absence of that cell rather than with anything
    the second constraint did.
    """
    p = _payload()
    s = p["summary"]
    matched = set(s["matched_density_targets"])
    a7_targets = {c["density_target"] for c in p["cells"]
                  if c["deployed_density"] is not None}
    ctrl_targets = {float(k) for k, v in p["controls"].items()
                    if v["deployed_density"] is not None}
    assert matched == a7_targets & ctrl_targets, (
        f"matched set {matched} is not the intersection of A7 {a7_targets} and controls "
        f"{ctrl_targets}"
    )
    assert "density_mae_A7" not in s, (
        "an unmatched MAE field is present; the confounded comparison has come back"
    )


def test_regime_s_survives_a_second_regime_e_constraint():
    """Validity must stay exactly 1.0 in every cell and control.

    The mask is enforced by construction, so anything below 1.0 means adding a second
    expectation constraint broke a guarantee that does not depend on the projection.
    """
    p = _payload()
    vals = {c["validity"] for c in p["cells"]} | {
        v["validity"] for v in p["controls"].values()}
    assert vals == {1.0}, f"structural validity is not exactly 1.0 everywhere: {vals}"


def test_fallback_rate_rises_with_the_second_target():
    """The competition mechanism, not just the outcome.

    If joint feasibility stopped collapsing as the second target moves away from what
    density implies, the explanation in the notes would no longer hold even if the
    pass-through number happened to stay low.
    """
    p = _payload()
    for r in p["rows"]:
        if not r.get("complete"):
            continue
        fb = r["fallback_rates"]
        assert all(b >= a - 1e-9 for a, b in zip(fb, fb[1:])), (
            f"fallback rate is not monotone in the valence target at density "
            f"{r['density_target']}: {fb}"
        )
        assert r["corr_target_fallback"] > 0.8, r["corr_target_fallback"]
        nj = r["joint_positions"]
        assert nj[-1] < nj[0], (
            f"joint positions did not collapse as the target hardened: {nj}"
        )


def test_pass_through_is_computed_from_the_cells():
    """Recompute the headline quantity rather than trusting the stored field."""
    p = _payload()
    for r in p["rows"]:
        if not r.get("complete"):
            continue
        va = np.array(r["valence_achieved"], dtype=float)
        vt = np.array(r["valence_targets"], dtype=float)
        want = float(va.max() - va.min()) / float(vt.max() - vt.min())
        assert abs(want - r["pass_through"]) < 1e-9, (
            f"stored pass-through {r['pass_through']} != recomputed {want}"
        )


def test_notes_do_not_overclaim_the_incomplete_grid():
    """One row is complete; the notes must not read as if the grid were finished."""
    if not NOTES.exists():
        pytest.skip("WS5 notes absent")
    txt = NOTES.read_text()
    p = _payload()
    n_complete = sum(1 for r in p["rows"] if r.get("complete"))
    if n_complete < 3:
        assert ("are not" in txt and "ceiling" in txt), (
            "the notes must disclose that rows are missing while the grid is incomplete"
        )
        for bad in ["all three rows", "the full grid shows", "across all nine cells"]:
            assert bad not in txt, f"notes overclaim with {bad!r} on an incomplete grid"


def test_valence_reference_matches_cp7():
    """The data mean and sd quoted here must be CP-7's measured values."""
    p = _payload()
    cp7 = RESULTS / "CP7_second_channel.json"
    if not cp7.exists():
        pytest.skip("CP-7 artifact absent")
    st = json.loads(cp7.read_text())["payload"]["channel_statistics"]
    assert abs(p["data_reference"]["valence_sd"]
               - st["valence_electron_density"]["sd"]) < 5e-4
    assert abs(p["data_reference"]["valence_mean"]
               - st["valence_electron_density"]["mean"]) < 5e-4
