"""Tests for the equivalence claim behind amortization."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "development"
pytestmark = pytest.mark.skipif(
    not DEV.exists(),
    reason="development/ is untracked; run-notes and preregistration checks need the dev tree")

ART = ROOT / "results" / "equivalence.json"
REP = ROOT / "results" / "retargeting_seed_replication.json"


def _p():
    if not ART.exists():
        pytest.skip("equivalence artifact absent")
    return json.loads(ART.read_text())["payload"]


def test_equivalence_recomputes_from_the_raw_paired_values():
    """Every headline figure must fall out of the replication artifact, not be stored."""
    p = _p()
    rows = json.loads(REP.read_text())["payload"]["paired_vs_frontier"]
    d = np.array([r["retarget_mean_error"] - r["frontier_error"] for r in rows])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    assert abs(p["mean_paired_difference"] - d.mean()) < 1e-9
    assert abs(p["ci95_on_mean_difference"][0] - (d.mean() - tc * se)) < 1e-9
    assert abs(p["ci95_on_mean_difference"][1] - (d.mean() + tc * se)) < 1e-9
    mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * d.std(ddof=1) / np.sqrt(n)
    assert abs(p["power"]["min_detectable_effect_80pct"] - mde) < 1e-9


def test_margin_is_the_arms_own_seed_spread_not_a_chosen_number():
    """A margin picked after seeing the result is not a pre-specification."""
    p = _p()
    rows = json.loads(REP.read_text())["payload"]["paired_vs_frontier"]
    want = float(np.mean([r["retarget_sd"] for r in rows]))
    got = p["equivalence"]["margin"]
    assert abs(got - want) <= 5e-4, (
        f"margin {got} is not the arm's mean seed sd {want:.6f} to the pre-registered "
        f"precision; the margin was changed or the data moved"
    )
    assert got == round(want, 3), (
        f"margin {got} is not the 3-decimal rounding of the rule value {want:.6f}"
    )
    prereg = ROOT / "development" / "preregistration" / "equivalence_margin.md"
    m = re.search(r"EQUIVALENCE_MARGIN_G_PER_CM3\s*=\s*([0-9.]+)", prereg.read_text())
    assert m and float(m.group(1)) == got, (
        "the artifact margin disagrees with the pre-registered file"
    )


def test_conclusion_is_not_margin_sensitive():
    """If equivalence held only at a generous margin it would be an artifact of the margin."""
    p = _p()
    sens = p["equivalence"]["sensitivity_to_margin"]
    tightest = min(float(k) for k in sens)
    assert sens[f"{tightest:.2f}"] < 0.05, (
        f"equivalence fails at the tightest margin tested ({tightest}); the conclusion "
        f"depends on the margin and must be stated as such"
    )


def test_verdict_matches_the_test_outcome():
    p = _p()
    if p["equivalence"]["equivalent_at_margin"]:
        assert "Equivalence established" in p["verdict"]
        assert p["equivalence"]["tost_p"] < 0.05
    else:
        assert "NOT established" in p["verdict"]


def test_prose_claims_equivalence_only_while_the_test_supports_it():
    """If the test ever stops passing, the paper must stop saying 'equivalent'."""
    p = _p()
    for name in ("results.md", "abstract.md"):
        f = ROOT / "development" / "docs" / "paper" / name
        if not f.exists():
            continue
        txt = re.sub(r"\s+", " ", f.read_text())
        if not p["equivalence"]["equivalent_at_margin"]:
            assert "equivalent to seven" not in txt, (
                f"{name} claims equivalence but the TOST does not establish it"
            )
        else:
            assert f"{p['equivalence']['tost_p']:.4f}" in txt, (
                f"{name} should carry the TOST p-value that licenses the word 'equivalent'"
            )


def test_power_is_reported_not_just_the_null():
    """A null result without its detectable effect size is not interpretable."""
    p = _p()
    txt = re.sub(r"\s+", " ", (ROOT / "development" / "docs" / "paper" / "results.md").read_text())
    assert f"{p['power']['min_detectable_effect_80pct']:.3f}" in txt, (
        "the minimum detectable effect is missing from the results prose"
    )
