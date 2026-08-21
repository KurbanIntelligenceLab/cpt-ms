"""The projected arm's induced sequence distribution against the exponential tilt."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.synthetic.induced_distribution import induced_distribution
from scripts.synthetic.targets import kl, tilting_target
from scripts.synthetic.task import TaskConfig, enumerate_task

SMALL = TaskConfig(length=5, max_count=2)


@pytest.fixture(scope="module")
def task():
    return enumerate_task(SMALL)


def test_induced_is_a_distribution(task):
    p, _ = induced_distribution(task, apply_S=True)
    assert np.isclose(p.sum(), 1.0, atol=1e-12)
    assert bool((p >= 0).all())


def test_induced_respects_regime_S(task):
    """Every sequence in the support must satisfy the support constraint."""
    p, _ = induced_distribution(task, apply_S=True)
    assert float(p[~task.feasible_S].sum()) == 0.0


def test_induced_support_equals_tilt_support(task):
    """Greedy projection must not extinguish reachable sequences."""
    p, _ = induced_distribution(task, apply_S=True)
    q, _ = tilting_target(task, apply_S=True)
    assert int((p > 0).sum()) == int((q > 0).sum())


def test_tilt_is_kl_optimal_and_induced_is_not(task):
    """The direction of the gap, which is the substance of the comparison."""
    base = np.where(task.feasible_S, task.p_data, 0.0)
    base = base / base.sum()
    tilt, _ = tilting_target(task, apply_S=True)
    induced, _ = induced_distribution(task, apply_S=True)
    assert kl(induced, base) >= kl(tilt, base) - 1e-12


def test_running_budget_becomes_infeasible_only_late(task):
    """The mechanism, asserted rather than described."""
    _, diag = induced_distribution(task, apply_S=True)
    by_pos = diag["infeasible_by_position"]
    first = diag["first_infeasible_position"]
    assert first is not None and first > 0, "infeasibility must not start at t=0"
    assert by_pos["0"]["infeasible"] == 0
    last = str(SMALL.length - 1)
    assert by_pos[last]["infeasible"] > 0, "late steps must show infeasibility"
    rates = [
        by_pos[str(t)]["infeasible"] / max(by_pos[str(t)]["nodes"], 1)
        for t in range(SMALL.length)
    ]
    assert rates[-1] > rates[first], "infeasibility must worsen toward the end"


def test_multiplier_bracket_is_not_saturated_at_a_fixed_bound(task):
    """Regression: a fixed bisection bracket returns its own endpoint."""
    _, diag = induced_distribution(task, apply_S=True)
    m = diag["max_abs_multiplier"]
    assert m not in (1.0, 60.0), f"multiplier pinned at a bracket endpoint: {m}"


def test_saturation_counter_is_not_a_tautology(task):
    """A diagnostic must be able to report absence, or it measures nothing."""
    _, diag = induced_distribution(task, apply_S=True)
    n, inf, sat = (
        diag["nodes_visited"],
        diag["per_step_infeasible_count"],
        diag["per_step_saturated_count"],
    )
    assert inf + sat != n, (
        "saturated + infeasible exhausts every node: the saturation check is "
        "flagging unconditionally rather than detecting a boundary"
    )
    assert sat < n - inf, "saturation cannot cover every feasible node"


def test_induced_reduces_but_does_not_eliminate_constraint_error(task):
    """Per-step projection moves the mean toward target without reaching it."""
    phi = task.total_charge.astype(np.float64)
    base = np.where(task.feasible_S, task.p_data, 0.0)
    base = base / base.sum()
    induced, _ = induced_distribution(task, apply_S=True)
    target = SMALL.charge_target
    assert abs(float(induced @ phi) - target) < abs(float(base @ phi) - target)
