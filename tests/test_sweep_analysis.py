"""The compensation regression must recover a known answer before it reads real data.

The compensation hypothesis is decided by a slope. An estimator that cannot recover a
slope it was given, or a verdict rule that fires on a slope of zero, would turn an
analysis bug into a finding — so both are checked against constructed data where the
answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.crystal.analyze_sweep import _ols


def _verdict(fit: dict) -> bool:
    """The rule used in analyze_sweep: significant AND materially nonzero."""
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
    """Significance alone is not the criterion; the magnitude gate is load-bearing.

    With low noise a slope of 0.05 is estimated precisely enough to clear any
    t-threshold while being physically meaningless. If the magnitude gate were
    dropped, this case would be reported as compensation.
    """
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
    """Pins the design gap: inside the data bulk, rejection is not expensive.

    The efficiency leg of the frontier only bites where rejection is costly. If a
    sweep covers only bulk targets it cannot produce an efficiency result, and this
    test exists so that limitation is asserted rather than discovered late.
    """
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
    """The compensation test must not regress a quantity on itself plus a term.

    `backbone_offset_vs_data = backbone - data_mean` and
    `projection_pull = backbone - deployed` both carry the backbone density with a
    positive sign, so offset - pull == deployed - data_mean identically. Regressing
    one on the other therefore has an algebraic slope near 1 regardless of whether
    any compensation exists, and the first version of this test reported slope 0.585
    with t = 3.66 on that basis alone.

    The valid regressor is the TARGET, which shares no term with the offset.
    """
    rng = np.random.default_rng(7)
    backbone = rng.normal(9.0, 2.0, 40)
    deployed = rng.normal(5.0, 1.0, 40)
    data_mean = 6.679
    offset = backbone - data_mean
    pull = backbone - deployed
    # The identity holds for ANY inputs -- that is exactly why the regression is void.
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
    """A near-zero linear slope is not evidence of no relationship.

    The backbone offset varies as an inverted U in the target: the linear fit gives
    R^2 = 0.02 while the quadratic gives 0.83 on the same rows. Reporting "no
    relationship" from the linear test alone is the same class of error as
    regressing a quantity on itself, and this test makes the shape check mandatory
    by showing a synthetic inverted U that the linear test declares flat.
    """
    x = np.linspace(1.0, 20.0, 12)
    y = 4.0 - 0.04 * (x - 10.0) ** 2          # symmetric about 10, so no linear trend
    lin = _ols(x, y)
    _, r2q, vertex = _quad(x, y)
    assert abs(lin["slope"]) < 0.05 and lin["r_squared"] < 0.05, (
        "fixture must look flat to a linear fit"
    )
    assert r2q > 0.99, "the quadratic must recover the curvature the linear fit missed"
    assert abs(vertex - 10.0) < 0.2, f"vertex {vertex:.2f} should recover the true 10.0"


def test_quadratic_vertex_is_not_the_argmax_of_the_observations():
    """Peak VALUE and vertex LOCATION are different quantities.

    Conflating them puts the mechanism at the wrong place on the axis: the largest
    observed offset sits at one target while the fitted vertex sits elsewhere.
    """
    x = np.array([1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 13.0, 16.0, 20.0])
    y = np.array([1.408, 1.512, 2.355, 2.517, 3.743, 3.582, 3.119, 1.779, 1.124])
    _, r2q, vertex = _quad(x, y)
    assert r2q > 0.8, "these are the measured offsets; curvature should be strong"
    argmax_x = float(x[np.argmax(y)])
    assert abs(vertex - argmax_x) > 1.0, (
        f"vertex {vertex:.2f} and argmax {argmax_x:.2f} must not be reported as the "
        "same quantity"
    )
