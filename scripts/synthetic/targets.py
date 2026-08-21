"""Exact constrained targets for the synthetic task (WS-2).

The plan requires every distortion result to be reported against BOTH candidate
targets: the data distribution restricted to the feasible set, and the base
model's distribution conditioned on the constraints (GAD's target). Enumerating
the task makes both exact.

Computing them surfaces a third object that the plan's two-target framing does
not name, and that matters for the projected-vs-comparator comparison:

  conditioning target   q(x) proportional to p(x) * 1[phi(x) == c]
                        the HARD constraint. Every sample satisfies it.

  tilting target        q(x) proportional to p(x) * exp(lambda * phi(x)),
                        lambda chosen so E_q[phi] = c
                        the EXPECTATION constraint. This is the KL-optimal
                        distribution satisfying the constraint in expectation,
                        i.e. GDC's EBM target and what comparator fine-tunes toward.

These are different distributions and they are not close. Regime E is an
expectation constraint, so the tilting target is its sequence-level ideal; the
conditioning target belongs to Regime S and to hard-constrained decoding. A
distortion result reported against the wrong one of the two would flatter or
penalize the method for the wrong reason, so both are computed here and the
choice is made explicit at the call site rather than by default.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .task import EnumeratedTask


def _normalize_log(log_w: np.ndarray) -> np.ndarray:
    """Softmax over log-weights, guarding the all-infeasible case."""
    finite = np.isfinite(log_w)
    if not finite.any():
        raise ValueError("no feasible sequences: constraint set is empty")
    m = log_w[finite].max()
    w = np.where(finite, np.exp(log_w - m), 0.0)
    total = w.sum()
    if total <= 0:
        raise ValueError("degenerate normalization: total weight is zero")
    return w / total


def conditioning_target(
    task: EnumeratedTask,
    *,
    log_base: np.ndarray | None = None,
    apply_S: bool = True,
    charge_equals: int | None = None,
) -> np.ndarray:
    """Base distribution conditioned on the hard feasible set.

    With `log_base=None` this is the data distribution restricted to the
    feasible set (target (i)). Passing a trained model's log-probabilities gives
    GAD's target (ii). The two coincide only when the base equals the data
    distribution, which is exactly the caveat the plan calls out.
    """
    log_w = (task.log_p_data if log_base is None else log_base).astype(np.float64).copy()
    if apply_S:
        log_w = np.where(task.feasible_S, log_w, -np.inf)
    if charge_equals is not None:
        log_w = np.where(task.total_charge == charge_equals, log_w, -np.inf)
    return _normalize_log(log_w)


def solve_tilt(
    task: EnumeratedTask,
    *,
    log_base: np.ndarray | None = None,
    apply_S: bool = True,
    charge_target: float | None = None,
    bracket: tuple[float, float] = (-50.0, 50.0),
) -> float:
    """Find the multiplier lambda with E_q[phi] = charge_target under the tilt.

    E_q[phi] is strictly increasing in lambda whenever phi is non-degenerate, so
    a bracketed root solve is exact to machine precision. The bracket is widened
    only on demand; a target outside the support's range has no finite solution
    and raises rather than silently returning an endpoint.
    """
    cfg = task.config
    target = cfg.charge_target if charge_target is None else charge_target
    log_w0 = (task.log_p_data if log_base is None else log_base).astype(np.float64)
    if apply_S:
        log_w0 = np.where(task.feasible_S, log_w0, -np.inf)

    phi = task.total_charge.astype(np.float64)
    support = np.isfinite(log_w0)
    lo_phi, hi_phi = phi[support].min(), phi[support].max()
    if not (lo_phi <= target <= hi_phi):
        raise ValueError(
            f"charge target {target} outside attainable range [{lo_phi}, {hi_phi}]"
        )

    def mean_phi(lam: float) -> float:
        q = _normalize_log(log_w0 + lam * phi)
        return float(q @ phi)

    lo, hi = bracket
    f_lo, f_hi = mean_phi(lo) - target, mean_phi(hi) - target
    if f_lo * f_hi > 0:
        # Target sits at the edge of the attainable range; report it rather than
        # returning a bracket endpoint that silently misses the constraint.
        raise ValueError(
            f"no finite multiplier on [{lo}, {hi}]: "
            f"E[phi] spans [{mean_phi(lo):.6f}, {mean_phi(hi):.6f}], target {target}"
        )
    return float(brentq(lambda lam: mean_phi(lam) - target, lo, hi, xtol=1e-12, rtol=1e-14))


def tilting_target(
    task: EnumeratedTask,
    *,
    log_base: np.ndarray | None = None,
    apply_S: bool = True,
    charge_target: float | None = None,
) -> tuple[np.ndarray, float]:
    """KL-optimal distribution satisfying the expectation constraint.

    Returns the distribution and the multiplier that produces it. This is the
    sequence-level ideal for Regime E and the target comparator approximates.
    """
    lam = solve_tilt(
        task, log_base=log_base, apply_S=apply_S, charge_target=charge_target
    )
    log_w = (task.log_p_data if log_base is None else log_base).astype(np.float64)
    if apply_S:
        log_w = np.where(task.feasible_S, log_w, -np.inf)
    return _normalize_log(log_w + lam * task.total_charge.astype(np.float64)), lam


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats. Zero where p is zero; infinite where q alone is."""
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance."""
    return float(0.5 * np.abs(p - q).sum())


def violation_rate(task: EnumeratedTask, p: np.ndarray, *, charge_equals: int | None = None) -> float:
    """Probability mass on sequences violating the hard charge constraint.

    Reported for every Regime-E distribution including the exact tilting target,
    which is the concrete form of the plan's "satisfaction in expectation only"
    caveat: the optimal expectation-constrained distribution still puts mass on
    individually violating sequences.
    """
    c = task.config.charge_target if charge_equals is None else charge_equals
    return float(p[task.total_charge != c].sum())
