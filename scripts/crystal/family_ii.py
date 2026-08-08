"""Regime-E family (ii): per-step determinability of the density bound (WS-2).

The research plan hedges the ionic-scoping risk with a second Regime-E family —
composition-budget and density-bound constraints — asserted to be "well-defined
for every composition." That assertion is true at the SEQUENCE level. This module
asks the question the projection layer actually needs answered: is it well-defined
per STEP, as an incremental linear functional of the next-token distribution?

Density is

    rho = (1/V) * sum_e n_e * m_e

with V the cell volume, n_e the count of element e, and m_e its atomic mass. A
per-step constraint needs the running expectation of rho to be a linear function
of the next-token distribution given the prefix. It is not, for a structural
reason: rho is a RATIO of two quantities that are both emitted incrementally, and
the expectation of a ratio is not the ratio of expectations.

Two escapes exist, and they behave very differently on the two backbones:

  (a) Deferred linearization. Once V is determined by the prefix, rho becomes
      LINEAR in the remaining element counts, since 1/V is then a known constant.
      The constraint is exactly a linear functional from that position onward and
      contributes nothing before it.

  (b) Mass-budget reformulation. Constrain total mass M = sum_e n_e m_e to an
      interval [rho_lo * V, rho_hi * V]. Still requires V.

Both escapes require V to be prefix-determined. So the operative question is:
at what fraction of the sequence is V determined, on each backbone?

That is what this module measures, and the answer decides whether the hedge is
real or nominal.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.crystal.parser import ConstraintState, Field, IncrementalParser


@dataclass
class DeterminabilityProfile:
    """Where along a sequence each constraint family becomes usable."""

    backbone: str
    n_tokens: int
    first_determined_position: int | None
    usable_fraction: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "backbone": self.backbone,
            "n_tokens": self.n_tokens,
            "first_determined_position": self.first_determined_position,
            "usable_fraction": self.usable_fraction,
            "reason": self.reason,
        }


def volume_determined_at(tokens: list[str], backbone: str) -> int | None:
    """First token position at which the cell volume is a function of the prefix.

    Returns None if the volume is never determined (malformed or truncated
    stream). Position is 1-indexed over emitted tokens.
    """
    parser = IncrementalParser(backbone)
    for i, state in enumerate(parser.run(tokens)):
        if state.cell_volume is not None:
            return i
    return None


def family_ii_profile(tokens: list[str], backbone: str) -> DeterminabilityProfile:
    """Fraction of a sequence over which family (ii) is an exact per-step constraint.

    The density bound is inactive until the cell volume is prefix-determined,
    because before that point rho is a ratio whose denominator is unknown and no
    linear functional of the next-token distribution represents it.
    """
    n = len(tokens)
    pos = volume_determined_at(tokens, backbone)
    if pos is None:
        return DeterminabilityProfile(
            backbone, n, None, 0.0,
            "cell volume never determined: lattice block absent or incomplete",
        )
    usable = (n - pos) / n if n else 0.0
    return DeterminabilityProfile(
        backbone, n, pos, usable,
        "linear in remaining element counts once 1/V is a known constant",
    )


def family_i_profile(tokens: list[str], backbone: str) -> DeterminabilityProfile:
    """Same measurement for family (i), charge balance, as a comparison baseline.

    Charge balance depends only on element counts, so it is determined when the
    composition block closes — which on a structured stream is early.
    """
    n = len(tokens)
    parser = IncrementalParser(backbone)
    pos = None
    for i, state in enumerate(parser.run(tokens)):
        if state.composition_closed:
            pos = i
            break
    if pos is None:
        return DeterminabilityProfile(
            backbone, n, None, 0.0, "composition block never closed"
        )
    return DeterminabilityProfile(
        backbone, n, pos, (n - pos) / n if n else 0.0,
        "linear in element counts once the composition block closes",
    )
