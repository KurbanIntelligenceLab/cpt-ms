"""The paper's prose must agree with the artifacts and with itself.

A figure quoted in three sections is three chances to drift. These tests recompute the
load-bearing numbers from the artifacts and assert the prose still carries them, and they
assert every artifact the paper cites actually exists -- a citation to a file that was
renamed or never written is the failure mode that survives every other check.
"""
from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
SECTIONS = ["introduction.md", "results.md", "discussion.md", "scope.md",
            "method_results_walkthrough.md"]


def _text(name: str) -> str:
    p = PAPER / name
    if not p.exists():
        pytest.skip(f"{name} absent")
    return re.sub(r"\s+", " ", p.read_text())


def _payload(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} absent")
    return json.loads(p.read_text())["payload"]


def test_every_cited_artifact_exists():
    """A citation to a missing artifact is unverifiable by definition."""
    have = {p.name for p in RESULTS.glob("*.json")}
    missing = []
    for name in SECTIONS:
        p = PAPER / name
        if not p.exists():
            continue
        for ref in set(re.findall(r"`results/([A-Za-z0-9_*]+\.json)`", p.read_text())):
            if "*" in ref:
                if not any(fnmatch.fnmatch(h, ref) for h in have):
                    missing.append((name, ref))
            elif ref not in have:
                missing.append((name, ref))
    assert not missing, f"cited artifacts absent from results/: {missing}"


def test_amortization_figures_match_the_artifact():
    s = _payload("WS4_retargeting_seed_replication.json")["summary"]
    txt = _text("introduction.md") + " " + _text("results.md")
    for want in (f"{s['sub_ceiling_mae_mean']:.3f}",
                 f"{s['sub_ceiling_mae_sd']:.3f}",
                 f"{s['frontier_sub_ceiling_mae']:.3f}"):
        assert want in txt, f"amortization figure {want} missing from the prose"
    assert f"p = {s['wilcoxon_p']:.1f}" in txt


def test_composition_figures_match_the_artifact():
    """Every complete row's load-bearing figures must appear, not just the first row's.

    The prose quotes per-row pass-through, the endpoint joint-position counts, and the two
    mechanism correlations; the full per-cell series lives in the figure, so requiring all six
    fallback rates to be typed into prose would force noise rather than catch drift. An earlier
    version of this test read rows[0] only, which would have passed while a second row's
    numbers were absent or wrong.
    """
    p = _payload("WS5_two_constraint_grid.json")
    rows = [r for r in p["rows"] if r.get("complete")]
    if not rows:
        pytest.skip("no complete grid row")
    txt = _text("results.md")
    for r in rows:
        assert f"{r['pass_through'] * 100:.1f}%" in txt, (
            f"pass-through for density {r['density_target']} "
            f"({r['pass_through'] * 100:.1f}%) is missing from results.md"
        )
        nj = r["joint_positions"]
        for x in (nj[0], nj[-1]):
            assert str(x) in txt, (
                f"endpoint joint-position count {x} for density {r['density_target']} missing"
            )
        c = r["corr_target_joint_positions"]
        assert (f"{c:.4f}" in txt or f"{c:+.4f}" in txt), (
            f"the joint-position correlation {c:.4f} for density {r['density_target']} is "
            f"missing; it is the quantity that distinguishes the rows"
        )
    if len(rows) >= 2:
        assert "compatib" in txt.lower(), (
            "with two disagreeing rows the prose must attribute the difference to target "
            "compatibility rather than reporting one row's outcome"
        )


def test_expressibility_figures_match_the_artifact():
    e = _payload("CP7_second_channel.json")["charge_is_not_linear"]
    ch = _payload("CP7_second_channel.json")["channel_statistics"]
    txt = _text("results.md") + " " + _text("discussion.md")
    assert f"{e['recall_of_smact_balanced'] * 100:.1f}%" in txt or \
           f"{e['recall_of_smact_balanced'] * 100:.2f}%" in txt
    assert f"{ch['correlation']:.3f}" in txt


def test_no_timeline_or_scheduling_language():
    """House rule: the paper carries no timelines."""
    bad = ["Week 1", "week 1", "next week", "by Monday", "timeline", "Timeline",
           "days from now", "tomorrow"]
    for name in SECTIONS:
        p = PAPER / name
        if not p.exists():
            continue
        t = p.read_text()
        for b in bad:
            assert b not in t, f"{name} contains scheduling language {b!r}"


def test_negative_results_are_stated_not_buried():
    """The composition limit and the expressibility boundary must appear in the paper.

    Both are results a later edit could quietly drop in favour of the flattering subset,
    and both are load-bearing for the honesty of the exactness claim.
    """
    txt = _text("results.md") + " " + _text("discussion.md")
    assert "not monotone" in txt, "the non-monotone second dial is not stated"
    assert "existential" in txt, "the expressibility boundary is not stated"
    assert "does not follow" in txt or "It does not follow" in txt, (
        "the exactness/steerability distinction is not drawn"
    )


def test_ceiling_claim_is_soft_not_hard():
    """Replication narrowed this; the prose must not carry the pre-replication reading."""
    s = _payload("WS4_retargeting_seed_replication.json")["summary"]
    txt = _text("results.md")
    assert s["seed0_is_an_excursion"] is True
    for x in s["backbone_across_seeds"][1:]:
        assert f"{x:.3f}" in txt, (
            f"replication backbone {x:.3f} missing; the prose may carry only the "
            f"single-seed reading"
        )
    assert "soft rather than hard" in txt or "soft" in txt
