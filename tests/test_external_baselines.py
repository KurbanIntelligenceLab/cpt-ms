"""Tests for the external comparisons: best-of-n, the automaton argument, the margin.

Two of these guard results that run AGAINST the paper's preferred framing, which is exactly
the kind of finding a later edit quietly softens. Best-of-n beats the projection on accuracy
at most targets, and the DFA size objection was computed and refuted. Both must survive.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PREREG = ROOT / "preregistration" / "WS4_equivalence_margin.md"


def _load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        pytest.skip(f"{name} absent")
    return json.loads(p.read_text())["payload"]


def test_best_of_n_estimates_expectation_not_the_pool_best():
    """Reporting the single best draw in a large pool would overstate best-of-n.

    The artifact must show repeated subsampling, and the error at n=1 must be close to the
    pool's mean absolute deviation from target rather than near zero.
    """
    p = _load("WS6_best_of_n.json")
    for r in p["rows"]:
        assert r["trials"] > 1, "best-of-n must be estimated over repeated draws"
    n1 = [r["mean_best_error"] for r in p["rows"] if r["n"] == 1]
    assert min(n1) > 1.0, (
        f"n=1 error {min(n1):.3f} is implausibly low for an unsteered pool; this looks like "
        f"the pool minimum rather than an expectation over single draws"
    )


def test_best_of_n_error_falls_monotonically_with_n():
    """A selection baseline that did not improve with n would indicate a bug, not a result."""
    p = _load("WS6_best_of_n.json")
    by: dict[float, dict[int, float]] = {}
    for r in p["rows"]:
        by.setdefault(r["target"], {})[r["n"]] = r["mean_best_error"]
    for t, row in by.items():
        ns = sorted(row)
        errs = [row[n] for n in ns]
        assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])), (
            f"best-of-n error is not monotone in n at target {t}: {errs}"
        )


def test_the_diversity_cost_of_selection_is_recorded():
    """The accuracy-only comparison favours best-of-n; the diversity column is the honest
    counterweight and must not be dropped."""
    p = _load("WS6_best_of_n.json")
    assert "diversity" in p and p["diversity"], "diversity measurement missing"
    by: dict[float, dict[int, float]] = {}
    for r in p["diversity"]:
        by.setdefault(r["target"], {})[r["n"]] = r["unique_formula_fraction"]
    for t, row in by.items():
        ns = sorted(row)
        assert row[ns[-1]] < row[ns[0]], (
            f"at target {t} diversity does not fall with n ({row}); selection must "
            f"concentrate the accepted set or the mechanism claim is wrong"
        )


def test_best_of_n_result_is_reported_as_competitive_not_dismissed():
    """A reviewer will run this baseline. The report must say it wins on accuracy."""
    note = ROOT / "reports" / "WS6_external_baselines_notes.md"
    if not note.exists():
        pytest.skip("WS-6 note absent")
    txt = re.sub(r"\s+", " ", note.read_text())
    assert "beats the projection" in txt or "reaches the projection" in txt, (
        "the note must state that best-of-n is competitive on accuracy"
    )
    assert "dismissing it would be indefensible" in txt or "not optional" in txt


def test_automaton_size_argument_reports_its_own_refutation():
    """The size objection was computed and did not hold. That outcome must stay visible."""
    p = _load("CP8_automaton_size.json")
    assert p["size_argument_verdict"].startswith("REFUTED"), (
        "the automaton size argument was refuted by its own computation; a verdict "
        "claiming otherwise means the computation or the framing changed"
    )
    tight = [r for r in p["rows"] if not r["at_or_below_task_floor"]][-1]
    assert tight["states_over_model_params"] < 1.0, (
        "the automaton at expressible tolerance is smaller than the model; the paper must "
        "not argue intractability"
    )
    assert "per-instance" in p["consequence"] or "separate automaton" in p["consequence"]


def test_automaton_tolerances_below_the_serialization_floor_are_flagged():
    """Demanding a tolerance the serialization cannot express would inflate the state count."""
    p = _load("CP8_automaton_size.json")
    floor = p["vocabulary"]["discretization_floor_relative"]
    for r in p["rows"]:
        assert r["at_or_below_task_floor"] == (r["relative_tolerance"] < floor), (
            f"tolerance {r['relative_tolerance']} misclassified against floor {floor}"
        )


def test_margin_lives_in_the_preregistration_not_the_script():
    """A margin editable in the analysis script is not a pre-specification."""
    assert PREREG.exists(), "the equivalence margin must be pre-registered"
    m = re.search(r"EQUIVALENCE_MARGIN_G_PER_CM3\s*=\s*([0-9.]+)", PREREG.read_text())
    assert m, "the pre-registration must define the margin as a parseable value"
    p = _load("WS4_equivalence.json")
    assert abs(p["equivalence"]["margin"] - float(m.group(1))) < 1e-9, (
        "the artifact's margin does not match the pre-registered value"
    )
    src = (ROOT / "scripts" / "crystal" / "equivalence_test.py").read_text()
    assert "read_prereg_margin" in src, "the script must read the margin, not define it"


def test_wilcoxon_is_reported_with_its_floor():
    """At n=7 the smallest attainable two-sided p is 2/2^7; quoting p alone overstates it."""
    p = _load("WS4_equivalence.json")
    dt = p["difference_tests"]
    n = p["n_paired_targets"]
    assert abs(dt["wilcoxon_min_attainable_two_sided_p"] - 2.0 / (2 ** n)) < 1e-12
    assert "resolution" in dt["wilcoxon_interpretation"] or \
           "smallest attainable" in dt["wilcoxon_interpretation"]


def test_preregistered_decision_rule_is_two_part_and_satisfied():
    p = _load("WS4_equivalence.json")
    eq = p["equivalence"]
    assert "0.10" in eq["decision_rule"], "the second condition must name the tighter margin"
    if eq["equivalent_at_margin"]:
        assert eq["decision_rule_satisfied"], (
            "equivalence is claimed at the margin but the pre-registered second condition "
            "(p < 0.05 at a 0.10 margin) is not met"
        )


def test_crossing_counts_in_prose_match_the_artifact():
    """The n-to-match counts are easy to hardcode and were once wrong.

    A reviewer caught 'four targets' where the artifact gives five. These counts are now
    recomputed from the best-of-n rows against the published frontier and asserted against
    the prose, so a literal cannot drift from the data again.
    """
    bon = _load("WS6_best_of_n.json")
    front = RES / "WS3_steering_frontier_rows.json"
    if not front.exists():
        pytest.skip("frontier artifact absent")
    rows = json.loads(front.read_text())["payload"]["rows"]
    by_t: dict[float, list[float]] = {}
    for v in rows.values():
        by_t.setdefault(v["target"], []).append(v["deployed_density"])
    proj = {t: abs(sum(v) / len(v) - t) for t, v in by_t.items()}

    be: dict[float, dict[int, float]] = {}
    for r in bon["rows"]:
        be.setdefault(r["target"], {})[r["n"]] = r["mean_best_error"]

    ns = sorted(bon["n_values"])
    cross = {t: next((n for n in ns if be[t][n] <= proj[t]), None)
             for t in proj if t in be}
    n_any = sum(1 for v in cross.values() if v is not None)
    n_at_smallest_useful = sum(1 for v in cross.values() if v == 4)

    for name in ("reports/WS6_external_baselines_notes.md", "paper/results.md"):
        f = ROOT / name
        if not f.exists():
            continue
        txt = re.sub(r"\s+", " ", f.read_text())
        assert f"{n_any} of {len(cross)} targets" in txt, (
            f"{name} does not carry the recomputed crossing count "
            f"{n_any} of {len(cross)}"
        )
        if "n = 4" in txt:
            assert f"at {n_at_smallest_useful} targets within n = 4" in txt or \
                   f"at {n_at_smallest_useful} targets it needs only n = 4" in txt, (
                f"{name} states an n=4 crossing count that disagrees with the artifact's "
                f"{n_at_smallest_useful}"
            )
