"""A3's induced sequence distribution vs the exponential tilt (WS-1/WS-2).

The central question the plan's A3-vs-A6 contrast rests on, and the only setting
where it can be answered exactly rather than estimated.

A3 applies the projection at every step: at each position it takes the model's
next-token distribution and projects it onto a PER-STEP expectation constraint.
A6 targets the sequence-level exponential tilt directly. These are different
objects and the plan treats the comparison as open. Here it is closed by
enumeration.

The per-step constraint A3 enforces cannot be the sequence constraint itself —
that is not a function of one step. The natural per-step surrogate, and the one
the layer implements, is a running budget: at each position, constrain the
expected increment so that the expected total lands on target given what the
prefix has already spent. Concretely, with T steps and a prefix that has
accumulated s after t steps, the remaining T - t steps must contribute c - s in
expectation, so the per-step target is

    b_t = (c - s_t) / (T - t)

This is exactly the "bounding expected increments bounds the expected total"
argument, and the enumerated comparison shows what it costs.

Three distributions, all computed exactly over the 65,536 enumerable sequences:

  base       the data/model distribution, unconstrained
  tilt       the sequence-level KL-optimal distribution satisfying E[phi] = c
  induced    the distribution A3's greedy per-step projection actually produces

The result of interest is not whether induced == tilt (it will not be) but the
DIRECTION and SIZE of the gap, and whether induced satisfies the sequence
constraint at all. The second question matters more: a per-step surrogate that
misses the sequence target is a different failure from one that hits it
sub-optimally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.synthetic.targets import kl, tilting_target, tv, violation_rate
from scripts.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "scripts/synthetic/induced_distribution.py"


def _step_conditionals(task, log_base: np.ndarray) -> None:
    """Placeholder to document that conditionals are derived, not stored."""
    raise NotImplementedError


def induced_distribution(
    task,
    *,
    charge_target: float | None = None,
    apply_S: bool = True,
    fallback: str = "base",
) -> tuple[np.ndarray, dict]:
    """Exact sequence distribution produced by greedy per-step projection.

    Walks the prefix tree. At each node the model's conditional over next tokens
    is projected onto the running-budget constraint, and the projected
    conditionals are multiplied along each path. Because the task is enumerable
    the walk is exact — no sampling, no truncation.

    Returns the distribution and diagnostics including how often the per-step
    projection was infeasible, which is the interesting failure mode: a running
    budget can demand an expected increment outside the attainable range of the
    remaining alphabet.
    """
    cfg = task.config
    V, L = cfg.vocab_size, cfg.length
    charges = np.asarray(cfg.charges, dtype=np.float64)
    target = cfg.charge_target if charge_target is None else float(charge_target)

    trans = task.transition
    init = task.initial

    n_infeasible = 0
    n_saturated = 0
    n_nodes = 0
    max_abs_lambda = 0.0
    # Per-position counters. The running budget becomes unsatisfiable once the
    # remaining steps cannot absorb the accumulated imbalance, so WHERE the
    # failures fall is the substance of the result, not an aside.
    per_t = {t: {"nodes": 0, "infeasible": 0} for t in range(L)}

    def project_step(p: np.ndarray, b: float, allowed: np.ndarray,
                     t: int = -1) -> np.ndarray:
        """Project one conditional onto E[phi] = b over the allowed support.

        Closed-form exponential tilt in one multiplier, solved by bisection on
        the monotone mean. Falls back to the masked base distribution when b is
        outside the attainable range, which is recorded rather than hidden.
        """
        nonlocal n_infeasible, n_saturated, max_abs_lambda
        q0 = np.where(allowed, p, 0.0)
        tot = q0.sum()
        if tot <= 0:
            return q0
        q0 = q0 / tot
        phi = charges
        lo_att, hi_att = phi[allowed].min(), phi[allowed].max()
        if not (lo_att - 1e-12 <= b <= hi_att + 1e-12):
            n_infeasible += 1
            if t >= 0:
                per_t[t]["infeasible"] += 1
            if fallback == "base":
                # Abandon the step's constraint and emit the masked base
                # conditional. Preserves fidelity at the cost of target accuracy.
                return q0
            elif fallback == "clamp":
                # Push as far toward the unreachable target as the alphabet
                # allows. Recovers target accuracy at the cost of extra
                # divergence and further variance collapse.
                b = float(np.clip(b, lo_att, hi_att))
            else:
                raise ValueError(f"unknown fallback policy {fallback!r}")

        def mean_at(lam: float) -> float:
            w = q0 * np.exp(lam * phi)
            return float((w * phi).sum() / w.sum())

        # Expand the bracket until it contains the root instead of assuming a
        # fixed width. A saturated bracket returns its own endpoint, which looks
        # like a converged multiplier and silently biases every downstream
        # number — the same class of error as the flat-dual line-search stall.
        lo, hi = -1.0, 1.0
        for _ in range(60):
            if mean_at(lo) <= b <= mean_at(hi):
                break
            lo *= 2.0
            hi *= 2.0
        else:
            n_infeasible += 1
            return q0
        # Capture the bracket BEFORE bisection. Bisection reassigns lo and hi, so
        # after it they have both converged onto the root — comparing the solved
        # multiplier against the post-bisection bound is a tautology that flags
        # every node. The meaningful question is whether the root sat near the
        # edge of the bracket the expansion loop had to open.
        bracket = abs(hi)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mean_at(mid) < b:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        # A target at the very edge of the attainable range needs a large
        # multiplier, which shows up as a root close to the expanded bracket's
        # own boundary. Measured against the ORIGINAL bracket, not the collapsed
        # one.
        if bracket > 1.0 and abs(lam) > 0.9 * bracket:
            n_saturated += 1
        max_abs_lambda = max(max_abs_lambda, abs(lam))
        w = q0 * np.exp(lam * phi)
        return w / w.sum()

    # Walk the tree over (prefix counts, accumulated charge) implicitly by
    # recursing on the full prefix, which is affordable at 4^8.
    probs = np.zeros(task.sequences.shape[0], dtype=np.float64)
    seq_index = {tuple(int(x) for x in row): i for i, row in enumerate(task.sequences)}

    def allowed_mask(counts: np.ndarray) -> np.ndarray:
        if not apply_S:
            return np.ones(V, dtype=bool)
        return counts < cfg.max_count

    def recurse(prefix: tuple[int, ...], logp: float, acc: float, counts: np.ndarray) -> None:
        nonlocal n_nodes
        t = len(prefix)
        if t == L:
            probs[seq_index[prefix]] = np.exp(logp)
            return
        n_nodes += 1
        p = init if t == 0 else trans[prefix[-1]]
        allowed = allowed_mask(counts)
        if not allowed.any():
            return
        remaining = L - t
        b_t = (target - acc) / remaining
        per_t[t]["nodes"] += 1
        q = project_step(p, b_t, allowed, t)
        for v in range(V):
            if q[v] <= 0:
                continue
            c2 = counts.copy()
            c2[v] += 1
            recurse(prefix + (v,), logp + float(np.log(q[v])),
                    acc + float(charges[v]), c2)

    recurse((), 0.0, 0.0, np.zeros(V, dtype=np.int64))

    total = probs.sum()
    diag = {
        "fallback_policy": fallback,
        "mass_before_renormalization": float(total),
        "nodes_visited": n_nodes,
        "per_step_infeasible_count": n_infeasible,
        "per_step_saturated_count": n_saturated,
        "per_step_infeasible_fraction": n_infeasible / max(n_nodes, 1),
        "max_abs_multiplier": max_abs_lambda,
        "infeasible_by_position": {str(t): per_t[t] for t in range(L)},
        "first_infeasible_position": next(
            (t for t in range(L) if per_t[t]["infeasible"] > 0), None
        ),
    }
    if total > 0:
        probs = probs / total
    return probs, diag


def main(smoke: bool = False) -> Path:
    cfg = TaskConfig(length=4, max_count=2) if smoke else TaskConfig()
    task = enumerate_task(cfg)
    phi = task.total_charge.astype(np.float64)

    base = task.p_data.copy()
    if True:  # Regime S applies to every arm; stated explicitly rather than implied.
        base_S = np.where(task.feasible_S, base, 0.0)
        base_S = base_S / base_S.sum()

    tilt, lam_seq = tilting_target(task, apply_S=True)
    induced, diag = induced_distribution(task, apply_S=True)

    def describe(name: str, p: np.ndarray) -> dict:
        mean = float(p @ phi)
        return {
            "name": name,
            "mean_charge": mean,
            "var_charge": float(p @ (phi - mean) ** 2),
            "entropy_nats": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
            "support_size": int((p > 0).sum()),
            "hard_violation_rate": violation_rate(task, p),
            "kl_from_base_S_nats": kl(p, base_S),
            "tv_to_base_S": tv(p, base_S),
        }

    rows = {
        "base_S": describe("base_S", base_S),
        "tilt": describe("tilt", tilt) | {"sequence_multiplier": lam_seq},
        "induced": describe("induced", induced),
    }

    comparison = {
        "kl_induced_from_tilt_nats": kl(induced, tilt),
        "kl_tilt_from_induced_nats": kl(tilt, induced),
        "tv_induced_tilt": tv(induced, tilt),
        "induced_mean_charge_error": float(induced @ phi) - cfg.charge_target,
        "tilt_mean_charge_error": float(tilt @ phi) - cfg.charge_target,
        "excess_kl_of_induced_over_tilt_nats": kl(induced, base_S) - kl(tilt, base_S),
        "entropy_gap_nats": rows["tilt"]["entropy_nats"] - rows["induced"]["entropy_nats"],
    }

    payload = {
        "distributions": rows,
        "comparison": comparison,
        "per_step_diagnostics": diag,
        "interpretation_keys": {
            "constraint_hit": abs(comparison["induced_mean_charge_error"]) < 1e-9,
            "induced_is_suboptimal": comparison["excess_kl_of_induced_over_tilt_nats"] > 0,
        },
    }

    return emit(
        "WS1_induced_vs_tilt" + ("_smoke" if smoke else ""),
        payload,
        config=cfg.as_dict() | {"smoke": smoke},
        seeds=[cfg.markov_seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
