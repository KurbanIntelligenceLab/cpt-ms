"""Tests for the comparator comparator and the claim it supports.

The claim is unusual and therefore easy to overstate: the surrogate is not a worse
approximation to the tilt, it approximates something else that violates less per sequence.
That reading only holds if comparator actually reached its own target, so the first test pins that.
If comparator ever fails to converge, the comparison becomes "our method beats a broken baseline" and
must not be reported as a structural finding.

One test exists because of a mistake made while writing this up: the mask-only training summary
stores [mean, sd] pairs, not per-seed lists, and averaging over a pair silently produced a
violation rate of 0.31 where the truth is 0.62. Anything reading those arrays must index
rather than aggregate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.common.arms import normalize, pick

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "development"
pytestmark = pytest.mark.skipif(
    not DEV.exists(),
    reason="development/ is untracked; run-notes and preregistration checks need the dev tree")

RES = ROOT / "results"
NOTES = ROOT / "development" / "reports" / "a6_comparator_notes.md"


def _p(name: str) -> dict:
    f = RES / name
    if not f.exists():
        pytest.skip(f"{name} absent")
    return json.loads(f.read_text())["payload"]


def test_comparator_reached_its_target_so_the_comparison_is_fair():
    """The comparator must satisfy the expectation constraint within its own seed spread."""
    p = _p("a6_tilt_comparator.json")
    s = p["summary"]
    mean, sd = s["mean_charge"]["mean"], s["mean_charge"]["sd"]
    assert abs(mean) <= 3 * max(sd, 1e-6), (
        f"the comparator's mean charge {mean:+.4f} is more than 3 seed sds ({sd:.4f}) from the target; "
        f"the comparison would be against a baseline that did not converge, and the "
        f"structural reading of the violation gap does not hold"
    )
    assert s["kl_from_tilt"]["mean"] < 0.05, (
        f"the comparator is {s['kl_from_tilt']['mean']:.4f} nats from the tilt; it is supposed to be an "
        f"approximation OF the tilt, so a large gap makes it the wrong comparator"
    )


def test_comparator_inherits_the_tilts_violation_rate():
    """The comparator approximating the tilt must inherit its per-sequence violation, or it is not
    approximating the tilt in the way the argument assumes."""
    p = _p("a6_tilt_comparator.json")
    a6 = p["summary"]["hard_violation_rate"]["mean"]
    tilt = p["reference"]["tilt"]["hard_violation_rate"]
    assert abs(a6 - tilt) < 0.05, (
        f"the comparator violates at {a6:.4f} but the tilt it approximates violates at {tilt:.4f}; the "
        f"claim that the tilt's violation rate is structural rather than a training artefact "
        f"rests on these agreeing"
    )


def test_mean_sd_pairs_are_indexed_not_averaged():
    """The mask-only summary stores [mean, sd]; averaging the pair corrupts every figure.

    This is pinned because it happened: mean([0.6155, 0.0080]) = 0.3117 was reported as a
    violation rate before the pair structure was noticed.
    """
    p = _p("training_summary.json")
    for arm, d in p["arms"].items():
        for field, v in d.items():
            if not isinstance(v, list) or len(v) != 2:
                continue
            mean, sd = v
            if mean is None or sd is None or not np.isfinite([mean, sd]).all():
                continue
            assert abs(sd) <= abs(mean) + 1.0, (
                f"{arm}.{field} = {v} does not look like a [mean, sd] pair; if the schema "
                f"changed to per-seed values, every reader of this artifact must change too"
            )
    a3 = normalize(p["arms"])["projected"]["violation_rate"]
    if NOTES.exists():
        txt = NOTES.read_text()
        assert f"{a3[0]:.4f}" in txt, (
            f"the notes must quote projected's violation rate {a3[0]:.4f} (the mean), not a value "
            f"derived by averaging the [mean, sd] pair"
        )
        assert f"{float(np.mean(a3)):.4f}" not in txt, (
            f"the notes contain {float(np.mean(a3)):.4f}, which is the average of the "
            f"[mean, sd] pair rather than any real quantity"
        )


def test_concentration_is_measured_against_the_optimum_not_the_unconstrained_arm():
    """The claim is concentration relative to the SEQUENCE-LEVEL OPTIMUM.

    An earlier draft compared projected's charge variance against the unconstrained arm, which is a
    true statement answering a different question: any constraint concentrates relative to no
    constraint. The claim being made is that the projection concentrates where the optimum
    does not, so the comparison must name the optimum. A reviewer caught this; the test keeps
    it caught.
    """
    a2p = _p("training_summary.json")
    a6p = _p("a6_tilt_comparator.json")
    a3 = pick(a2p["arms"], "projected")["charge_variance"][0]
    a2 = pick(a2p["arms"], "mask_only")["charge_variance"][0]
    tilt = a6p["reference"]["tilt"]["charge_variance"]
    a6 = a6p["summary"]["charge_variance"]["mean"]
    a6_sd = a6p["summary"]["charge_variance"]["sd"]
    assert a3 < tilt, (
        f"the projected arm's charge variance {a3:.3f} is not below the optimum's {tilt:.3f}; the "
        f"concentration claim as stated does not hold"
    )
    assert a6 - 3 * a6_sd > a3, (
        f"the projected-vs-comparator variance gap is inside three comparator seed sds ({a6:.3f} +/- {a6_sd:.3f} "
        f"against {a3:.3f}); it must not be reported as established"
    )
    if NOTES.exists():
        txt = re.sub(r"\s+", " ", NOTES.read_text())
        assert f"{tilt / a3:.2f}" in txt, (
            f"the notes must state the factor against the OPTIMUM ({tilt / a3:.2f}), which is "
            f"the claim; the factor against the unconstrained arm ({a2 / a3:.1f}) is a "
            f"different and weaker statement"
        )
        assert f"{tilt:.3f}" in txt and f"{a6:.3f}" in txt, (
            "both the exact optimum's and comparator's variance must appear; a blank cell there was "
            "the original defect"
        )


def test_the_optimum_anti_concentrates_relative_to_the_data():
    """The optimum's variance EXCEEDS the distribution it started from.

    This is the mechanism, and it is stronger than 'the optimum does not concentrate'. If a
    later run reversed it, the prose would be wrong in a way no other test would catch.
    """
    a6p = _p("a6_tilt_comparator.json")
    tilt = a6p["reference"]["tilt"]["charge_variance"]
    base = a6p["reference"]["base_S_conditioned"]["charge_variance"]
    assert tilt > base, (
        f"the optimum's charge variance {tilt:.3f} does not exceed the S-conditioned data's "
        f"{base:.3f}; the anti-concentration statement in the prose is unsupported"
    )
    if NOTES.exists():
        txt = re.sub(r"\s+", " ", NOTES.read_text())
        assert f"{100 * (tilt / base - 1):.1f}%" in txt, (
            "the notes must quote how far above the data the optimum sits"
        )


def test_the_claim_is_not_stated_as_beating_the_baseline_on_its_own_terms():
    """The projected arm does not approximate the tilt better than comparator does; it approximates something else.

    Overstating this as 'the surrogate is a better tilt approximation' would be false, since
    comparator's KL from the tilt is far smaller than the induced distribution's.
    """
    if not NOTES.exists():
        pytest.skip("notes absent")
    txt = re.sub(r"\s+", " ", NOTES.read_text())
    assert "not a worse approximation to the tilt" in txt or "different thing" in txt, (
        "the notes must frame the result as approximating a different target, not as a "
        "better approximation of the same one"
    )
    a6_kl = _p("a6_tilt_comparator.json")["summary"]["kl_from_tilt"]["mean"]
    induced_kl = _p("induced_vs_tilt.json")["comparison"]["kl_induced_from_tilt_nats"]
    assert a6_kl < induced_kl, (
        f"the comparator ({a6_kl:.4f} nats) should be closer to the tilt than the per-step induced "
        f"distribution ({induced_kl:.4f}); if not, the framing above is wrong"
    )
