"""Actionability: where the projection can still MOVE a constraint (WS-2).

Determinability and actionability are different questions and they can be
complements rather than overlaps:

  determinable at t   the constraint's value is computable from the prefix
  actionable at t     a different token at t would change the constraint's
                      final value

A per-step projection needs BOTH. A constraint that is determined but frozen
admits no projection — there is nothing left to steer. This module measures
actionability by counterfactual substitution rather than by reasoning about the
format, because the two can disagree and the format argument is the one that has
already been wrong once here.

Why this matters for CP0. The gate reads the fraction of positions with an active
multiplier at convergence. If a constraint is only actionable over a handful of
positions, a correctly functioning layer produces a small sequence-averaged
activity, and a threshold calibrated against the determinable fraction would fail
a working layer. The threshold has to be set on activity WITHIN the actionable
window.

Method. For each position t, substitute every candidate token and ask whether the
final constraint value changes. A position where no substitution changes it is
frozen: determined or not, the projection cannot act there. This is expensive but
exact, and these sequences are short enough for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from scripts.crystal.parser import IncrementalParser

# Charges used for the family (i) statistic. Values are conventional and only the
# fact that they DIFFER between elements matters for actionability.
CHARGES = {"Na": 1, "Cl": -1, "Mg": 2, "O": -2, "Fe": 3}
MASSES = {"Na": 22.99, "Cl": 35.45, "Mg": 24.31, "O": 16.00, "Fe": 55.85}


@dataclass
class ActionabilityProfile:
    backbone: str
    family: str
    n_tokens: int
    n_actionable: int
    actionable_fraction: float
    first_actionable: int | None
    last_actionable: int | None
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _final_state(tokens: list[str], backbone: str):
    return list(IncrementalParser(backbone).run(tokens))[-1]


def _charge(state) -> float | None:
    """Total formal charge, or None while composition is open."""
    if not state.composition_closed:
        return None
    return float(sum(CHARGES.get(el, 0) * n for el, n in state.element_counts.items()))


def _density(state) -> float | None:
    """Mass density proxy, or None while composition or lattice is open."""
    v = state.cell_volume
    if v is None or not state.composition_closed:
        return None
    mass = sum(MASSES.get(el, 0.0) * n for el, n in state.element_counts.items())
    return mass / v


STAT = {"i_charge": _charge, "ii_density": _density}


def _candidates(tokens: list[str], t: int, backbone: str) -> list[str]:
    """Plausible alternative tokens at position t.

    Restricted to substitutions the format could actually produce, so the measure
    reports freedom the model really has rather than freedom an arbitrary token
    swap would invent.
    """
    tok = tokens[t]
    if backbone == "B2":
        if tok.startswith("el:"):
            _, el, cnt = tok.split(":")
            return [f"el:{o}:{cnt}" for o in CHARGES if o != el] + \
                   [f"el:{el}:{int(cnt)+1}"]
        if tok.startswith("lat:"):
            _, key, val = tok.split(":")
            return [f"lat:{key}:{float(val)*1.1:.4f}"]
        return []
    # Character backbones: only digits and element letters carry meaning for
    # these two statistics.
    if tok.isdigit():
        return [str((int(tok) + 1) % 10)]
    if tok.isalpha():
        return [c for c in "NCMOFabcegl" if c != tok][:3]
    return []


def profile(tokens: list[str], backbone: str, family: str) -> ActionabilityProfile:
    """Count positions where a substitution changes the final constraint value."""
    stat = STAT[family]
    baseline = stat(_final_state(tokens, backbone))
    n = len(tokens)

    actionable: list[int] = []
    for t in range(n):
        for cand in _candidates(tokens, t, backbone):
            alt = list(tokens)
            alt[t] = cand
            try:
                val = stat(_final_state(alt, backbone))
            except Exception:
                # A substitution that makes the stream unparseable is not
                # evidence of steering freedom; skip rather than count it.
                continue
            if val is None and baseline is None:
                continue
            if val is None or baseline is None or abs(val - baseline) > 1e-12:
                actionable.append(t)
                break

    if not actionable:
        return ActionabilityProfile(
            backbone, family, n, 0, 0.0, None, None,
            "no substitution changes the constraint: determined but frozen",
        )
    return ActionabilityProfile(
        backbone, family, n, len(actionable), len(actionable) / n,
        actionable[0], actionable[-1],
        "positions where a format-legal substitution moves the constraint",
    )
