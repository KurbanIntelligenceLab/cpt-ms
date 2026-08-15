"""Tests for the SMC baseline, and for the honesty of how it is reported.

Two things need pinning. The density SMC steers must be the density the projection targets, or
the two methods are compared on different quantities. And the reporting must stay honest in a
specific way: a best-per-target width selected on the same runs that report the error is
selection on the evaluation set, so that number must never become the headline.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "results" / "WS6_smc_steering.json"
NOTES = ROOT / "reports" / "WS6b_smc_notes.md"


def _p() -> dict:
    if not ART.exists():
        pytest.skip("SMC artifact absent")
    return json.loads(ART.read_text())["payload"]


def test_smc_scores_the_same_density_the_projection_targets():
    """The SMC potential's density estimate must equal the published decoder exactly.

    A second parse could disagree about bucket centres, and the two methods would then be
    compared on different densities without anything looking wrong.
    """
    from scripts.crystal.mp20 import ELEMENTS, CrystalVocab, load_split
    from scripts.crystal.smc_steering import _density_estimate

    V = CrystalVocab(ELEMENTS)
    train = load_split("train", limit=40)
    enc = [V.encode(c) for c in train]
    width = max(len(e) for e in enc)
    arr = np.array([e + [V.index["<EOS>"]] * (width - len(e)) for e in enc])
    est = _density_estimate(arr, V, None)
    dec = np.array([V.decode_density(e) for e in enc], dtype=float)
    ok = np.isfinite(est) & np.isfinite(dec)
    assert ok.sum() >= 20, f"only {ok.sum()} comparable structures"
    assert np.allclose(est[ok], dec[ok], atol=1e-9), (
        f"the SMC density estimate diverges from decode_density by up to "
        f"{np.abs(est[ok] - dec[ok]).max():.3e}; the baseline would be steering a different "
        f"quantity than the projection targets"
    )


def test_validity_is_exact_at_every_particle_count():
    """The mask is active during SMC, so support exactness must not depend on K."""
    p = _p()
    for r in p["rows"]:
        assert r["n_valid"] == r["runs"], (
            f"K={r['K']} target {r['target']}: {r['n_valid']}/{r['runs']} valid; the mask "
            f"should make every SMC output structurally valid"
        )


def test_error_falls_with_particle_count_on_average():
    """More particles must help on average, or the implementation is not doing SMC."""
    p = _p()
    by_k: dict[int, list[float]] = {}
    for r in p["rows"]:
        if r["mean_error"] is not None and r.get("sigma", 1.0) == 1.0:
            by_k.setdefault(r["K"], []).append(r["mean_error"])
    ks = sorted(by_k)
    if len(ks) < 2:
        pytest.skip("fewer than two particle counts")
    means = [float(np.mean(by_k[k])) for k in ks]
    assert means[-1] < means[0], (
        f"mean error did not fall from K={ks[0]} to K={ks[-1]}: {means}"
    )


def test_the_tuned_number_is_not_reported_as_smc_performance():
    """Selection on the evaluation set must be disclosed, not used.

    The best-per-target width gives a better mean than any fixed width. If that number ever
    becomes the reported result, this test fails.
    """
    p = _p()
    sweep = p.get("sigma_sweep")
    if not sweep:
        pytest.skip("no width sweep recorded")
    assert "why_no_tuned_headline" in sweep, (
        "the width sweep must record why the best-per-target number is not the headline"
    )
    assert "selection on the evaluation set" in sweep["why_no_tuned_headline"]
    if NOTES.exists():
        txt = re.sub(r"\s+", " ", NOTES.read_text())
        by_t: dict[float, dict[float, float]] = {}
        for r in sweep["rows"]:
            if r["mean_error"] is not None:
                by_t.setdefault(r["target"], {})[r["sigma"]] = r["mean_error"]
        best = float(np.mean([min(v.values()) for v in by_t.values()]))
        assert f"{best:.3f}" in txt, (
            "the best-per-target mean must be stated, so a reader can see what was declined"
        )
        assert "not reported as SMC's performance" in txt or "not reported as SMC" in txt, (
            "the notes must say the tuned number is not the reported performance"
        )


def test_the_untuned_status_is_stated_as_a_bound():
    p = _p()
    sweep = p.get("sigma_sweep") or {}
    left = sweep.get("left_untuned") or p.get("lower_bound_note") or ""
    assert "proposal" in left and "resampling" in left, (
        "the untuned components must be named, since they bound what SMC could achieve"
    )
    if NOTES.exists():
        txt = re.sub(r"\s+", " ", NOTES.read_text())
        assert "lower bound" in txt, (
            "the notes must state that the SMC figures are a lower bound rather than its "
            "ceiling; otherwise the comparison reads as a refutation of SMC"
        )
