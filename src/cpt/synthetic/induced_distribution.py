"""The projected arm's induced sequence distribution vs the exponential tilt."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cpt.common.artifacts import emit
from cpt.synthetic.targets import kl, tilting_target, tv, violation_rate
from cpt.synthetic.task import TaskConfig, enumerate_task

SCRIPT = "src/cpt/synthetic/induced_distribution.py"


def _step_conditionals(task, log_base: np.ndarray) -> None:
    raise NotImplementedError


def induced_distribution(
    task,
    *,
    charge_target: float | None = None,
    apply_S: bool = True,
    fallback: str = "base",
) -> tuple[np.ndarray, dict]:
    """Exact sequence distribution produced by greedy per-step projection."""
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
    per_t = {t: {"nodes": 0, "infeasible": 0} for t in range(L)}

    def project_step(p: np.ndarray, b: float, allowed: np.ndarray,
                     t: int = -1) -> np.ndarray:
        """Project one conditional onto E[phi] = b over the allowed support."""
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
                return q0
            if fallback == "clamp":
                b = float(np.clip(b, lo_att, hi_att))
            else:
                raise ValueError(f"unknown fallback policy {fallback!r}")

        def mean_at(lam: float) -> float:
            w = q0 * np.exp(lam * phi)
            return float((w * phi).sum() / w.sum())

        lo, hi = -1.0, 1.0
        for _ in range(60):
            if mean_at(lo) <= b <= mean_at(hi):
                break
            lo *= 2.0
            hi *= 2.0
        else:
            n_infeasible += 1
            return q0
        bracket = abs(hi)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mean_at(mid) < b:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        if bracket > 1.0 and abs(lam) > 0.9 * bracket:
            n_saturated += 1
        max_abs_lambda = max(max_abs_lambda, abs(lam))
        w = q0 * np.exp(lam * phi)
        return w / w.sum()

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
            recurse((*prefix, v), logp + float(np.log(q[v])),
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
    if True:
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
            "mass_outside_S": float(p[~task.feasible_S].sum()),
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
        "induced_vs_tilt" + ("_smoke" if smoke else ""),
        payload,
        config=cfg.as_dict() | {"smoke": smoke},
        seeds=[cfg.markov_seed],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
