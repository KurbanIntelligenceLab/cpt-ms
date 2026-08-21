"""The compensation regression must recover a known answer before it reads real data."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.crystal.analyze_sweep import _ols

from _devtree import requires_mp20

pytestmark = requires_mp20


def _verdict(fit: dict) -> bool:
    return fit["slope_t_vs_zero"] > 2.0 and fit["slope"] > 0.3


@pytest.mark.parametrize("slope,intercept", [(1.0, 0.0), (0.6, -0.4), (0.0, 2.5)])
def test_ols_recovers_a_known_slope(slope, intercept):
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 12)
    y = intercept + slope * x + rng.normal(0, 0.05, len(x))
    fit = _ols(x, y)
    assert abs(fit["slope"] - slope) < 0.05, f"slope {fit['slope']:.3f} vs {slope}"
    assert abs(fit["intercept"] - intercept) < 0.05


def test_verdict_rejects_a_flat_slope():
    """No compensation must read as no compensation."""
    rng = np.random.default_rng(1)
    x = np.linspace(-3, 3, 12)
    fit = _ols(x, 2.5 + rng.normal(0, 0.05, len(x)))
    assert not _verdict(fit), "a flat offset was reported as compensation"


def test_verdict_rejects_a_significant_but_tiny_slope():
    """Significance alone is not the criterion; the magnitude gate is load-bearing."""
    rng = np.random.default_rng(2)
    x = np.linspace(-3, 3, 12)
    fit = _ols(x, 0.05 * x + rng.normal(0, 0.01, len(x)))
    assert fit["slope_t_vs_zero"] > 2.0, "fixture should be statistically significant"
    assert not _verdict(fit), "a negligible slope passed as compensation"


def test_verdict_accepts_genuine_compensation():
    """A slope near one must be detected, or the test cannot find the effect."""
    rng = np.random.default_rng(3)
    x = np.linspace(-3, 3, 12)
    fit = _ols(x, 1.0 * x + rng.normal(0, 0.05, len(x)))
    assert _verdict(fit)
    assert abs(fit["slope_t_vs_one"]) < 3.0, "slope should be consistent with 1"


def test_slope_stderr_is_reported_so_near_one_can_be_judged():
    """A slope without an uncertainty cannot support 'near 1'."""
    rng = np.random.default_rng(4)
    x = np.linspace(-3, 3, 12)
    fit = _ols(x, x + rng.normal(0, 0.05, len(x)))
    assert np.isfinite(fit["slope_stderr"]) and fit["slope_stderr"] > 0
    assert fit["dof"] == len(x) - 2


def test_rejection_cost_matches_a_direct_count():
    """The efficiency baseline must be the data's own acceptance rate, not a model."""
    from scripts.crystal.analyze_sweep import rejection_cost
    from scripts.crystal.mp20 import load_split
    d = np.array([c.density for c in load_split("train")])
    for t in (3.0, 5.0, 10.0):
        r = rejection_cost(t, tol=0.5)
        direct = float(((d > t - 0.5) & (d < t + 0.5)).mean())
        assert abs(r["acceptance_rate"] - direct) < 1e-12
        assert abs(r["expected_draws"] - 1.0 / direct) < 1e-9


def test_bulk_targets_are_cheap_for_rejection():
    """Pins the design gap: inside the data bulk, rejection is not expensive."""
    from scripts.crystal.analyze_sweep import rejection_cost
    for t in (3.0, 5.0, 6.317, 8.0):
        assert rejection_cost(t)["expected_draws"] < 25, (
            f"target {t} was assumed cheap for rejection but is not"
        )
    for t in (1.5, 16.0):
        assert rejection_cost(t)["expected_draws"] > 50, (
            f"tail target {t} should be expensive for rejection"
        )


def test_offset_and_pull_share_a_term_so_their_regression_is_invalid():
    """The compensation test must not regress a quantity on itself plus a term."""
    rng = np.random.default_rng(7)
    backbone = rng.normal(9.0, 2.0, 40)
    deployed = rng.normal(5.0, 1.0, 40)
    data_mean = 6.679
    offset = backbone - data_mean
    pull = backbone - deployed
    assert np.abs((offset - pull) - (deployed - data_mean)).max() < 1e-12
    fit = _ols(pull, offset)
    assert fit["slope"] > 0.3 and fit["slope_t_vs_zero"] > 2.0, (
        "purely random inputs must still produce a 'supported' verdict, which is the "
        "proof that this regression cannot test compensation"
    )


def _quad(x, y):
    X = np.column_stack([np.ones(len(x)), x, x ** 2])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    r2 = 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return b, r2, (-b[1] / (2 * b[2]) if b[2] else float("nan"))


def test_a_linear_fit_cannot_rule_out_a_non_monotone_relationship():
    """A near-zero linear slope is not evidence of no relationship."""
    x = np.linspace(1.0, 20.0, 12)
    y = 4.0 - 0.04 * (x - 10.0) ** 2
    lin = _ols(x, y)
    _, r2q, vertex = _quad(x, y)
    assert abs(lin["slope"]) < 0.05 and lin["r_squared"] < 0.05, (
        "fixture must look flat to a linear fit"
    )
    assert r2q > 0.99, "the quadratic must recover the curvature the linear fit missed"
    assert abs(vertex - 10.0) < 0.2, f"vertex {vertex:.2f} should recover the true 10.0"


def test_quadratic_vertex_is_not_the_argmax_of_the_observations():
    """Peak VALUE and vertex LOCATION are different quantities."""
    x = np.array([1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 13.0, 16.0, 20.0])
    y = np.array([1.408, 1.512, 2.355, 2.517, 3.743, 3.582, 3.119, 1.779, 1.124])
    _, r2q, vertex = _quad(x, y)
    assert r2q > 0.8, "these are the measured offsets; curvature should be strong"
    argmax_x = float(x[np.argmax(y)])
    assert abs(vertex - argmax_x) > 1.0, (
        f"vertex {vertex:.2f} and argmax {argmax_x:.2f} must not be reported as the "
        "same quantity"
    )
