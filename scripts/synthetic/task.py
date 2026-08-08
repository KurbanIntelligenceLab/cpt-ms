"""Synthetic constrained-sequence task (WS-2).

The only setting in the project where the constrained target is computable
exactly rather than estimated. Everything here is brute-force enumerable by
construction, so the "ground truth" is arithmetic, not a fit.

Design mirrors the crystal setting in miniature:

  Regime S (support restriction)   a running multiplicity budget — each symbol
                                   may appear at most `max_count` times. Prefix-
                                   closed, so it is exactly a per-step mask,
                                   which is what makes A2 representable.

  Regime E (expectation constraint) a charge-balance analogue — each symbol
                                   carries an integer charge and the constraint
                                   is on the expected total. Linear in the
                                   distribution, satisfied in expectation only,
                                   with per-sample violations that the exact
                                   target does not eliminate. This is the point.

The base/data distribution is a first-order Markov chain, deliberately biased so
that it does NOT already satisfy the charge constraint. If it did, the tilt
would be zero and every target would collapse onto the data distribution — the
degenerate case, and the one worth being able to switch on and off when probing
multiplier decay.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    """Full specification of the synthetic task. Hashed into every artifact."""

    symbols: tuple[str, ...] = ("A", "B", "C", "D")
    charges: tuple[int, ...] = (2, 1, -1, -2)
    length: int = 8
    max_count: int = 3          # Regime S: per-symbol occurrence budget
    charge_target: float = 0.0  # Regime E: constraint on E[total charge]
    markov_seed: int = 0
    markov_bias: float = 1.6    # >1 tilts the chain toward positive charge
    markov_temperature: float = 0.8

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnumeratedTask:
    """The task, fully enumerated. All arrays are indexed by sequence id."""

    config: TaskConfig
    sequences: np.ndarray      # (N, L) int8 token ids
    log_p_data: np.ndarray     # (N,) log probability under the data distribution
    total_charge: np.ndarray   # (N,) int  — the Regime-E statistic phi(x)
    feasible_S: np.ndarray     # (N,) bool — satisfies the multiplicity budget
    transition: np.ndarray = field(repr=False, default=None)  # (V, V)
    initial: np.ndarray = field(repr=False, default=None)     # (V,)

    @property
    def p_data(self) -> np.ndarray:
        return np.exp(self.log_p_data)


def build_markov(config: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    """Seeded first-order Markov chain over the vocabulary.

    The charge bias is applied as a multiplicative tilt on the emission of each
    symbol, which is what pushes E[total charge] away from the target and makes
    the Regime-E constraint bind.
    """
    rng = np.random.default_rng(config.markov_seed)
    V = config.vocab_size
    charges = np.asarray(config.charges, dtype=float)

    logits = rng.normal(0.0, 1.0, size=(V, V)) * config.markov_temperature
    logits = logits + np.log(config.markov_bias) * charges[None, :]
    transition = np.exp(logits)
    transition /= transition.sum(axis=1, keepdims=True)

    init_logits = rng.normal(0.0, 1.0, size=V) * config.markov_temperature
    init_logits = init_logits + np.log(config.markov_bias) * charges
    initial = np.exp(init_logits)
    initial /= initial.sum()

    return transition, initial


def enumerate_task(config: TaskConfig | None = None) -> EnumeratedTask:
    """Enumerate every sequence and compute its data probability and statistics."""
    config = config or TaskConfig()
    V, L = config.vocab_size, config.length
    transition, initial = build_markov(config)

    sequences = np.array(list(itertools.product(range(V), repeat=L)), dtype=np.int8)
    n = sequences.shape[0]
    assert n == V ** L, f"enumeration size {n} != {V**L}"

    log_init = np.log(initial)
    log_trans = np.log(transition)
    log_p = log_init[sequences[:, 0]].astype(np.float64)
    for t in range(1, L):
        log_p += log_trans[sequences[:, t - 1], sequences[:, t]]

    charges = np.asarray(config.charges, dtype=np.int64)
    total_charge = charges[sequences].sum(axis=1)

    counts = np.zeros((n, V), dtype=np.int16)
    for v in range(V):
        counts[:, v] = (sequences == v).sum(axis=1)
    feasible_S = (counts <= config.max_count).all(axis=1)

    return EnumeratedTask(
        config=config,
        sequences=sequences,
        log_p_data=log_p,
        total_charge=total_charge,
        feasible_S=feasible_S,
        transition=transition,
        initial=initial,
    )


def prefix_allowed_mask(config: TaskConfig, prefix: tuple[int, ...]) -> np.ndarray:
    """Regime-S mask: which next tokens keep the multiplicity budget satisfiable.

    Prefix-closed by construction — a prefix that has already spent a symbol's
    budget can never recover it — which is exactly the property that makes the
    mask emitter well-defined and A2 trainable without straight-through.
    """
    V = config.vocab_size
    counts = np.zeros(V, dtype=np.int64)
    for tok in prefix:
        counts[tok] += 1
    return counts < config.max_count
