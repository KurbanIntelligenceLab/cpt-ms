"""WS7 charge marginals must agree with the published multi-seed summaries and must
exhibit the paper's anti-concentration ordering.

results/charge_marginals.json retrains mask-only, projected, and comparator at a single seed to recover
exact per-charge-value marginals that were never stored (only scalar summaries were,
in results/training_summary.json and results/tilt_comparator.json). This file
checks that the single-seed marginals' derived scalars are consistent with those
published multi-seed summaries, and that the headline mechanism -- the per-step
projection CONCENTRATES mass near the constraint target while the sequence-level tilt
SPREADS mass toward both extremes to hit the same mean -- is visible in the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.common.arms import normalize, pick

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "charge_marginals.json"
A2_SUMMARY = ROOT / "results" / "training_summary.json"
A6_SUMMARY = ROOT / "results" / "tilt_comparator.json"


def _payload(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    return json.loads(path.read_text())["payload"]


def test_marginals_are_normalized_and_cover_the_full_support():
    p = _payload(ARTIFACT)
    for name, d in p["distributions"].items():
        masses = np.array(d["masses"], dtype=float)
        assert masses.min() >= -1e-12, f"{name} has negative mass"
        assert abs(masses.sum() - 1.0) < 1e-6, f"{name} marginal does not sum to 1"
        assert len(d["values"]) == len(d["masses"])


def test_agreement_check_recorded_and_within_bounds():
    """Every scalar the script itself checked against published multi-seed values
    must be recorded as within a plausible seed spread (4 published sd)."""
    p = _payload(ARTIFACT)
    checks = p.get("agreement_check_vs_published")
    assert checks, "no agreement_check_vs_published recorded in the artifact"
    bad = [c for c in checks if not c["within_4sd"]]
    assert not bad, f"single-seed values fall outside plausible seed spread: {bad}"


def test_agreement_check_matches_published_artifacts_directly():
    """Independently recompute the same comparison from the published summary
    artifacts, rather than trusting the recorded check alone."""
    p = _payload(ARTIFACT)
    a2 = normalize(_payload(A2_SUMMARY)["arms"])
    a6 = _payload(A6_SUMMARY)

    dist = normalize(p["distributions"])

    def within(single, mean, sd, k=4.0):
        if sd == 0.0:
            return abs(single - mean) < 0.5
        return abs(single - mean) <= k * sd

    assert within(pick(dist, "mask_only")["mean_charge"],
                  pick(a2, "mask_only")["mean_charge"][0], pick(a2, "mask_only")["mean_charge"][1])
    assert within(pick(dist, "mask_only")["charge_variance"],
                  pick(a2, "mask_only")["charge_variance"][0], pick(a2, "mask_only")["charge_variance"][1])
    assert within(dist["projected"]["mean_charge"],
                  a2["projected"]["mean_charge"][0], a2["projected"]["mean_charge"][1])
    assert within(dist["projected"]["charge_variance"],
                  a2["projected"]["charge_variance"][0], a2["projected"]["charge_variance"][1])
    assert within(dist["comparator"]["mean_charge"],
                  a6["summary"]["mean_charge"]["mean"], a6["summary"]["mean_charge"]["sd"])
    assert within(dist["comparator"]["charge_variance"],
                  a6["summary"]["charge_variance"]["mean"], a6["summary"]["charge_variance"]["sd"])
    # Exact (no-seed) objects must match to machine precision.
    assert dist["tilt"]["charge_variance"] == pytest.approx(
        a6["reference"]["tilt"]["charge_variance"], abs=1e-9)
    assert dist["data_S_conditioned"]["charge_variance"] == pytest.approx(
        a6["reference"]["base_S_conditioned"]["charge_variance"], abs=1e-9)


def test_anti_concentration_ordering_holds():
    """The paper's headline mechanism: projection concentrates below the data's own
    spread, while the sequence-level tilt (and comparator, which targets it) spread wider
    than the data to buy a mean of zero."""
    p = _payload(ARTIFACT)
    dist = normalize(p["distributions"])
    var_proj = dist["projected"]["charge_variance"]
    var_data = dist["data_S_conditioned"]["charge_variance"]
    var_tilt = dist["tilt"]["charge_variance"]
    var_a6 = dist["comparator"]["charge_variance"]

    assert var_proj < var_data, "projection should concentrate BELOW the data's own spread"
    assert var_data < var_tilt, "the exact tilt should spread WIDER than the data"
    assert var_a6 > var_data, "the comparator, trained toward the tilt, should also spread wider than the data"


def test_a3_deployed_mean_charge_near_target():
    """The projected arm should land close to the charge_target=0 constraint."""
    p = _payload(ARTIFACT)
    mean = normalize(p["distributions"])["projected"]["mean_charge"]
    assert abs(mean) < 0.1, f"projected mean charge {mean} is far from the target 0.0"


def test_single_seed_caveat_is_disclosed():
    """The artifact must not silently present a single seed as the published multi-seed
    result."""
    p = _payload(ARTIFACT)
    assert "single_seed_caveat" in p, "single-seed caveat must be recorded in the artifact"
    assert p["config"]["seed"] == 0 if "config" in p else True
