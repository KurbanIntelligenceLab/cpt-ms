"""Regime-E family (ii): per-step determinability of the density bound."""

from __future__ import annotations

from dataclasses import dataclass

from cpt.crystal.parser import IncrementalParser


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
    """First token position at which the cell volume is a function of the prefix."""
    parser = IncrementalParser(backbone)
    for i, state in enumerate(parser.run(tokens)):
        if state.cell_volume is not None:
            return i
    return None


def family_ii_profile(tokens: list[str], backbone: str) -> DeterminabilityProfile:
    """Fraction of a sequence over which family (ii) is an exact per-step constraint."""
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
    """Same measurement for family (i), charge balance, as a comparison baseline."""
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
