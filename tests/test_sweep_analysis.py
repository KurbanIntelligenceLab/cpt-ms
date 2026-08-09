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
