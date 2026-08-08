"""The steering frontier and the fallback-policy trade.

The positive capability claim rests on three legs measured together, so the tests
check that each leg is real and that the relationship between them is not an
artifact of picking one flattering target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.synthetic.induced_distribution import induced_distribution
from scripts.synthetic.steering_frontier import frontier
from scripts.synthetic.targets import kl, tilting_target
from scripts.synthetic.task import TaskConfig, enumerate_task

SMALL = TaskConfig(length=5, max_count=2)


@pytest.fixture(scope="module")
def task():
    return enumerate_task(SMALL)


@pytest.fixture(scope="module")
def base(task):
    b = np.where(task.feasible_S, task.p_data, 0.0)
    return b / b.sum()


def _targets(task):
    """Targets spanning the attainable range INCLUDING its extremes.

    The extremes are load-bearing: they are exactly where the running budget goes
    badly infeasible, where the induced distribution fails to reach the requested
    target, and therefore where the at-target divergence comparison breaks down.
    A fixture that trims them would leave the comparator defect untested.
    """
    att = sorted({int(x) for x in np.unique(task.total_charge[task.feasible_S])})
    mid = [t for t in att if abs(t) <= 2]
    return sorted({att[0], att[-1], *mid})


def test_rejection_baseline_cost_is_the_reciprocal_acceptance(task, base):
    """The efficiency baseline must be the exact reciprocal, not an estimate."""
    rows = frontier(task, _targets(task))
    phi = task.total_charge.astype(np.float64)
    for r in rows:
        if not r["target_reachable_by_rejection"]:
            continue
        mass = float(base[np.isclose(phi, r["charge_target"])].sum())
        assert r["rejection_acceptance_rate"] == pytest.approx(mass, abs=1e-15)
        assert r["rejection_draws_per_sample"] == pytest.approx(1.0 / mass, rel=1e-12)


def test_projection_enriches_the_target_set(task):
    """The capability claim: A3 concentrates mass where the base does not.

    Enrichment below 1 would mean the projection moved mass AWAY from the target,
    which would invalidate the whole steering framing.
    """
    rows = frontier(task, _targets(task))
    enrich = [r["target_mass_enrichment"] for r in rows
              if r["target_mass_enrichment"] is not None]
    assert enrich, "no reachable target produced an enrichment figure"
    assert all(e > 1.0 for e in enrich), f"projection failed to enrich: {enrich}"


def test_excess_divergence_is_never_negative(task, base):
    """The tilt is optimal, so no per-step route can beat it at equal constraint.

    Regression for a real defect. Comparing the induced distribution against the
    tilt AT THE REQUESTED TARGET produced negative excess at extreme targets,
    which would read as the surrogate beating a provable optimum. The cause was
    that the two were not solving the same problem: where the running budget goes
    badly infeasible the induced distribution does not reach the target, so it is
    not paying that constraint. Compared against the tilt at the mean it ACTUALLY
    achieves, both satisfy the same constraint and the difference is a genuine
    cost — non-negative by construction.
    """
    rows = frontier(task, _targets(task))
    checked = 0
    for r in rows:
        if r["excess_kl_over_tilt_at_achieved_nats"] is not None:
            assert r["excess_kl_over_tilt_at_achieved_nats"] >= -1e-9, r["charge_target"]
            checked += 1
        # The at-target comparator is only reported where it is valid.
        if r["excess_kl_over_tilt_at_target_nats"] is not None:
            assert r["excess_comparison_valid"]
            assert r["excess_kl_over_tilt_at_target_nats"] >= -1e-9, r["charge_target"]
    assert checked == len(rows), "achieved comparator must be defined at every target"


def test_achieved_comparator_covers_targets_the_at_target_one_cannot(task):
    """The achieved comparator must not silently inherit the same blind spot.

    If both comparators were valid on exactly the same rows there would be no
    reason to have two, and the strong-steering end of the frontier — where the
    capability claim lives — would remain uncovered.
    """
    rows = frontier(task, _targets(task))
    at_target = sum(1 for r in rows if r["excess_kl_over_tilt_at_target_nats"] is not None)
    achieved = sum(1 for r in rows if r["excess_kl_over_tilt_at_achieved_nats"] is not None)
    assert achieved > at_target, (
        f"achieved comparator covers {achieved} rows, at-target covers {at_target}: "
        "the achieved comparator is meant to extend coverage"
    )


def test_waste_fraction_falls_as_steering_strengthens(task):
    """The headline relationship, checked per side of the base mean.

    Excess divergence is roughly a fixed overhead while the necessary divergence
    scales with steering strength, so the wasted fraction must fall as the target
    moves away from the base. Checked separately either side, because the
    attainable range is not centred on the base mean.
    """
    rows = [r for r in frontier(task, _targets(task))
            if r["waste_fraction_of_induced_divergence"] is not None]
    for sign in (-1, +1):
        side = [r for r in rows if np.sign(r["signed_steering_strength"]) == sign]
        if len(side) < 2:
            continue
        side.sort(key=lambda r: abs(r["signed_steering_strength"]))
        w = [r["waste_fraction_of_induced_divergence"] for r in side]
        assert all(b <= a + 1e-9 for a, b in zip(w, w[1:])), f"sign {sign}: {w}"


def test_fallback_policies_trade_accuracy_against_divergence(task, base):
    """Both policies are defensible and the trade must be visible, not asserted.

    Clamping to the nearest attainable increment recovers target accuracy at the
    cost of extra divergence; falling back to the base conditional does the
    reverse. A pre-registration that picks one needs this measured.
    """
    phi = task.total_charge.astype(np.float64)
    tilt, _ = tilting_target(task, apply_S=True)
    kt = kl(tilt, base)

    p_base, d_base = induced_distribution(task, apply_S=True, fallback="base")
    p_clamp, d_clamp = induced_distribution(task, apply_S=True, fallback="clamp")

    err_base = abs(float(p_base @ phi) - SMALL.charge_target)
    err_clamp = abs(float(p_clamp @ phi) - SMALL.charge_target)
    assert err_clamp < err_base, "clamping must improve target accuracy"
    assert kl(p_clamp, base) > kl(p_base, base), "clamping must cost divergence"
    assert kl(p_base, base) >= kt - 1e-12 and kl(p_clamp, base) >= kt - 1e-12
    assert d_base["fallback_policy"] == "base"
    assert d_clamp["fallback_policy"] == "clamp"


def test_unknown_fallback_policy_is_rejected(task):
    """A silently-ignored policy name would make the pre-registration unenforceable."""
    with pytest.raises(ValueError, match="unknown fallback"):
        induced_distribution(task, apply_S=True, fallback="redistribute")
