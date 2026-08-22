"""Tests for the comparator comparator and the claim it supports."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from _artifacts import payload
from cpt.common.arms import normalize, pick

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "development"
pytestmark = pytest.mark.skipif(
    not DEV.exists(),
    reason="development/ is untracked; run-notes and preregistration checks need the dev tree")

RES = ROOT / "results"
NOTES = ROOT / "development" / "reports" / "a6_comparator_notes.md"


def _p(name: str) -> dict:
    return payload(name)


def test_comparator_reached_its_target_so_the_comparison_is_fair():
    """The comparator must satisfy the expectation constraint within its own seed spread."""
    p = _p("tilt_comparator.json")
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
    """Approximating the tilt means inheriting its per-sequence violation rate."""
    p = _p("tilt_comparator.json")
    comparator = p["summary"]["hard_violation_rate"]["mean"]
    tilt = p["reference"]["tilt"]["hard_violation_rate"]
    assert abs(comparator - tilt) < 0.05, (
        f"the comparator violates at {comparator:.4f} but the tilt it approximates violates at {tilt:.4f}; the "
        f"claim that the tilt's violation rate is structural rather than a training artefact "
        f"rests on these agreeing"
    )


def test_mean_sd_pairs_are_indexed_not_averaged():
    """The mask-only summary stores [mean, sd]; averaging the pair corrupts every figure."""
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
    projected = normalize(p["arms"])["projected"]["violation_rate"]
    if NOTES.exists():
        txt = NOTES.read_text()
        assert f"{projected[0]:.4f}" in txt, (
            f"the notes must quote projected's violation rate {projected[0]:.4f} (the mean), not a value "
            f"derived by averaging the [mean, sd] pair"
        )
        assert f"{float(np.mean(projected)):.4f}" not in txt, (
            f"the notes contain {float(np.mean(projected)):.4f}, which is the average of the "
            f"[mean, sd] pair rather than any real quantity"
        )


def test_concentration_is_measured_against_the_optimum_not_the_unconstrained_arm():
    """The claim is concentration relative to the SEQUENCE-LEVEL OPTIMUM."""
    mask_only_payload = _p("training_summary.json")
    comparator_payload = _p("tilt_comparator.json")
    projected = pick(mask_only_payload["arms"], "projected")["charge_variance"][0]
    mask_only = pick(mask_only_payload["arms"], "mask_only")["charge_variance"][0]
    tilt = comparator_payload["reference"]["tilt"]["charge_variance"]
    comparator = comparator_payload["summary"]["charge_variance"]["mean"]
    comparator_sd = comparator_payload["summary"]["charge_variance"]["sd"]
    assert projected < tilt, (
        f"the projected arm's charge variance {projected:.3f} is not below the optimum's {tilt:.3f}; the "
        f"concentration claim as stated does not hold"
    )
    assert comparator - 3 * comparator_sd > projected, (
        f"the projected-vs-comparator variance gap is inside three comparator seed sds ({comparator:.3f} +/- {comparator_sd:.3f} "
        f"against {projected:.3f}); it must not be reported as established"
    )
    if NOTES.exists():
        txt = re.sub(r"\s+", " ", NOTES.read_text())
        assert f"{tilt / projected:.2f}" in txt, (
            f"the notes must state the factor against the OPTIMUM ({tilt / projected:.2f}), which is "
            f"the claim; the factor against the unconstrained arm ({mask_only / projected:.1f}) is a "
            f"different and weaker statement"
        )
        assert f"{tilt:.3f}" in txt and f"{comparator:.3f}" in txt, (
            "both the exact optimum's and comparator's variance must appear; a blank cell there was "
            "the original defect"
        )


def test_the_optimum_anti_concentrates_relative_to_the_data():
    """The optimum's variance EXCEEDS the distribution it started from."""
    comparator_payload = _p("tilt_comparator.json")
    tilt = comparator_payload["reference"]["tilt"]["charge_variance"]
    base = comparator_payload["reference"]["base_S_conditioned"]["charge_variance"]
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
    """The projected arm approximates a different distribution, not the tilt better."""
    if not NOTES.exists():
        pytest.skip("notes absent")
    txt = re.sub(r"\s+", " ", NOTES.read_text())
    assert "not a worse approximation to the tilt" in txt or "different thing" in txt, (
        "the notes must frame the result as approximating a different target, not as a "
        "better approximation of the same one"
    )
    comparator_kl = _p("tilt_comparator.json")["summary"]["kl_from_tilt"]["mean"]
    induced_kl = _p("induced_vs_tilt.json")["comparison"]["kl_induced_from_tilt_nats"]
    assert comparator_kl < induced_kl, (
        f"the comparator ({comparator_kl:.4f} nats) should be closer to the tilt than the per-step induced "
        f"distribution ({induced_kl:.4f}); if not, the framing above is wrong"
    )
