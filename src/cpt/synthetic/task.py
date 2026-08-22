"""Synthetic constrained-sequence task."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    """Full specification of the synthetic task. Hashed into every artifact."""

    symbols: tuple[str, ...] = ("A", "B", "C", "D")
    charges: tuple[int, ...] = (2, 1, -1, -2)
    length: int = 8
    max_count: int = 3
    charge_target: float = 0.0
    markov_seed: int = 0
    markov_bias: float = 1.6
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
    sequences: np.ndarray
    log_p_data: np.ndarray
    total_charge: np.ndarray
    feasible_S: np.ndarray
    transition: np.ndarray = field(repr=False, default=None)
    initial: np.ndarray = field(repr=False, default=None)

    @property
    def p_data(self) -> np.ndarray:
        return np.exp(self.log_p_data)


def build_markov(config: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    """Seeded first-order Markov chain over the vocabulary."""
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
    """Regime-S mask: which next tokens keep the multiplicity budget satisfiable."""
    V = config.vocab_size
    counts = np.zeros(V, dtype=np.int64)
    for tok in prefix:
        counts[tok] += 1
    return counts < config.max_count
