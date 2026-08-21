"""Tests for the A7 two-constraint grid and its analysis.

The WS-5 conclusion is a NEGATIVE result, which makes it exactly the kind of claim that
rots quietly: if a later change makes the second dial steer, the report must stop saying it
does not. These tests pin the criteria the verdict is derived from, so the verdict cannot
survive a change in the underlying measurement.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"
ART = RESULTS / "two_constraint_grid.json"
NOTES = Path(__file__).resolve().parents[1] / "reports" / "two_constraint_grid_notes.md"


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

    Fallback must rise with the second target in EVERY row: a harder second ask means more
    positions where no distribution satisfies both budgets, regardless of compatibility.
    This is the part of the mechanism that does not depend on which row you look at.
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


def test_joint_position_direction_tracks_target_compatibility():
    """The finding the second row produced, pinned so it cannot be flattened back.

    Joint positions do NOT simply collapse with the second target. They fall where the two
    targets conflict (low density wanting light or few atoms, high valence wanting many
    electrons per volume) and rise where they agree. An earlier version of this test asserted
    the collapse unconditionally, which was the one-row reading; asserting the
    compatibility-dependent direction is what the completed rows support.
    """
    p = _payload()
    rows = {r["density_target"]: r for r in p["rows"] if r.get("complete")}
    if len(rows) < 2:
        pytest.skip("fewer than two complete rows")

    # The conflicting row is the one whose density target is furthest BELOW the data mean;
    # the compatible row is above it. Derived from the artifact, not hardcoded.
    mean = p["data_reference"]["density_mean"] if "density_mean" in p.get(
        "data_reference", {}) else 6.679
    below = [t for t in rows if t < mean]
    above = [t for t in rows if t >= mean]
    if not below or not above:
        pytest.skip("no row on each side of the data mean")

    lo, hi = rows[max(below)], rows[min(above)]
    assert lo["corr_target_joint_positions"] < -0.5, (
        f"at density {lo['density_target']} (below the data mean, so conflicting with a high "
        f"valence ask) joint positions should FALL with the second target; correlation is "
        f"{lo['corr_target_joint_positions']}"
    )
    assert hi["corr_target_joint_positions"] > 0.5, (
        f"at density {hi['density_target']} (above the data mean, so compatible) joint "
        f"positions should RISE with the second target; correlation is "
        f"{hi['corr_target_joint_positions']}"
    )
    assert hi["pass_through"] > lo["pass_through"], (
        f"the compatible row must pass through more of the request than the conflicting one: "
        f"{hi['pass_through']:.4f} vs {lo['pass_through']:.4f}"
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
    n_rows_planned = 3
    if n_complete < n_rows_planned:
        # The disclosure must state the count, not merely gesture at incompleteness.
        words = {1: "One", 2: "Two", 3: "Three"}
        forms = [f"{n_complete} rows of {n_rows_planned}",
                 f"{n_complete} of {n_rows_planned}",
                 f"{words.get(n_complete, n_complete)} of three",
                 f"{words.get(n_complete, n_complete)} complete rows",
                 "not yet complete"]
        assert any(f in txt for f in forms), (
            f"{n_complete} of {n_rows_planned} rows are complete; the notes must say so "
            f"rather than reading as a finished grid"
        )
        for bad in ["all three rows", "the full grid shows", "across all nine cells"]:
            assert bad not in txt, f"notes overclaim with {bad!r} on an incomplete grid"
    if n_complete >= 2:
        # With two rows the single-row conclusion is falsified and must not be ASSERTED.
        # A sentence that explicitly disavows it ("Not 'they do not compose' -- that was the
        # one-row reading") is the correct way to retire a claim, so only unquoted assertions
        # count as a violation.
        for line in txt.splitlines():
            if "do not compose" not in line:
                continue
            disavowed = ('Not "' in line or "Not '" in line or "not \u201c" in line.lower()
                         or "\u201c" in line or "one-row" in line)
            assert disavowed, (
                f"two complete rows disagree, so this line asserts the retired one-row "
                f"claim: {line.strip()[:100]!r}"
            )


def test_valence_reference_matches_cp7():
    """The data mean and sd quoted here must be CP-7's measured values."""
    p = _payload()
    cp7 = RESULTS / "second_channel.json"
    if not cp7.exists():
        pytest.skip("CP-7 artifact absent")
    st = json.loads(cp7.read_text())["payload"]["channel_statistics"]
    assert abs(p["data_reference"]["valence_sd"]
               - st["valence_electron_density"]["sd"]) < 5e-4
    assert abs(p["data_reference"]["valence_mean"]
               - st["valence_electron_density"]["mean"]) < 5e-4


def test_fixed_valence_columns_use_partial_rows_correctly():
    """A partial row is valid for the direction analysis and invalid for pass-through.

    Pass-through is a range over valence targets, so one cell cannot produce it. The
    deviation-from-control comparison at a shared valence target across density rows needs only
    one cell per row. This test pins both halves: the partial row must be absent from
    pass-through and present in the columns.
    """
    p = _payload()
    cols = p["summary"].get("fixed_valence_columns")
    if not cols:
        pytest.skip("no fixed-valence columns")
    partial = [r for r in p["rows"] if not r.get("complete")]
    for r in partial:
        assert "pass_through" not in r, (
            f"density {r['density_target']} has {r['n_cells']} cell(s) but carries a "
            f"pass-through value, which needs a range over valence targets"
        )
        dts_in_cols = {row["density_target"] for c in cols for row in c["rows"]}
        if r["n_cells"] >= 1:
            assert r["density_target"] in dts_in_cols, (
                f"density {r['density_target']} is partial but contributes to no column; its "
                f"cell is usable for the direction comparison and is being discarded"
            )


def test_direction_is_measured_against_each_cells_own_control():
    """Deviation must be from the matching single-constraint control, not a global mean.

    The achieved valence rises with the density target even with no valence constraint, so an
    absolute level would show apparent steering where there is none.
    """
    p = _payload()
    cols = p["summary"].get("fixed_valence_columns") or []
    if not cols:
        pytest.skip("no fixed-valence columns")
    controls = p["controls"]
    for c in cols:
        for r in c["rows"]:
            ctrl = (controls.get(str(r["density_target"]))
                    or controls.get(r["density_target"]) or {})
            assert abs(r["control_valence"] - ctrl["deployed_valence"]) < 1e-12, (
                f"column at valence {c['valence_target']}, density {r['density_target']} "
                f"compares against {r['control_valence']} but that density's control is "
                f"{ctrl['deployed_valence']}"
            )
            want = r["valence_achieved"] - r["control_valence"]
            assert abs(r["deviation_from_control"] - want) < 1e-12


def test_the_directional_asymmetry_is_reported_if_present():
    """Lowering the quantity is always affordable; raising it is not once the budget is spent.

    If this asymmetry is in the data it must be in the notes, because it is a mechanism claim a
    reader can check and a later edit could quietly drop.
    """
    p = _payload()
    cols = p["summary"].get("fixed_valence_columns") or []
    if not cols:
        pytest.skip("no fixed-valence columns")
    rows = [r for c in cols for r in c["rows"]]
    down = [r for r in rows if r["asked_direction"] == "down"]
    up = [r for r in rows if r["asked_direction"] == "up"]
    if not down or not up:
        pytest.skip("only one requested direction present")
    d_ok = sum(r["moved_correct_direction"] for r in down) / len(down)
    u_ok = sum(r["moved_correct_direction"] for r in up) / len(up)
    if d_ok > u_ok and NOTES.exists():
        txt = re.sub(r"\s+", " ", NOTES.read_text())
        assert f"{sum(r['moved_correct_direction'] for r in down)} of {len(down)}" in txt, (
            "the direction tally must be stated; it is the sharpest mechanism evidence in the "
            "grid and does not depend on completing any row"
        )
        assert "exhaust" in txt.lower(), (
            "the notes must connect the asymmetry to budget exhaustion, which is the "
            "mechanism that explains it"
        )
