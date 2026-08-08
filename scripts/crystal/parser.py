"""Incremental constraint-state parser and mask emitter (WS-2).

The projection layer needs, at every decoding step, a constraint state that is a
function of the prefix alone. This module defines that state for the two
backbones and reports which constraint families are *determined* at which token
positions.

Two backbones, two very different determinability profiles:

  B1  CIF-character stream. Tokens are characters of a CIF file. Composition and
      lattice parameters appear at scattered positions and are not known until
      their fields have been emitted.

  B2  Structured field stream. Tokens are typed fields in a fixed order
      (composition block, then lattice, then sites). Field boundaries are known
      by construction.

The distinction that matters for the research plan is between a constraint that
is *prefix-closed* (its feasibility can only shrink as the prefix grows, so a
per-step mask is exact) and one that is merely *prefix-measurable* (its value is
known but the constraint is on a total that is not yet complete).

Regime S (composition budget) is prefix-closed on both backbones.
Regime E family (i) (charge balance) is prefix-measurable once the composition
    block closes.
Regime E family (ii) (density bound) is the hard case and is analyzed in
    `family_ii.py`, not here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class Field(Enum):
    """CIF-ish field regions, in the order a structured stream emits them."""

    PRELUDE = "prelude"
    COMPOSITION = "composition"
    LATTICE = "lattice"
    SITES = "sites"
    DONE = "done"


@dataclass
class ConstraintState:
    """Everything the projection layer may condition on at one decoding step.

    Every field here is a pure function of the prefix. Nothing is imputed from
    the future, and nothing is estimated: a quantity that is not yet determined
    is None, never a guess. Downstream code must branch on None rather than
    treating an absent value as zero.
    """

    position: int = 0
    region: Field = Field.PRELUDE
    element_counts: dict[str, int] = field(default_factory=dict)
    composition_closed: bool = False
    lattice_params: dict[str, float] = field(default_factory=dict)
    lattice_closed: bool = False
    sites_emitted: int = 0

    @property
    def n_atoms(self) -> int | None:
        """Total atom count, or None while the composition block is open."""
        return sum(self.element_counts.values()) if self.composition_closed else None

    @property
    def cell_volume(self) -> float | None:
        """Cell volume in cubic angstroms, or None while the lattice is open.

        General triclinic formula; the orthogonal case falls out of it.
        """
        if not self.lattice_closed:
            return None
        p = self.lattice_params
        try:
            a, b, c = p["a"], p["b"], p["c"]
            al, be, ga = (math.radians(p[k]) for k in ("alpha", "beta", "gamma"))
        except KeyError:
            return None
        t = (
            1
            - math.cos(al) ** 2
            - math.cos(be) ** 2
            - math.cos(ga) ** 2
            + 2 * math.cos(al) * math.cos(be) * math.cos(ga)
        )
        return a * b * c * math.sqrt(t) if t > 0 else None

    _buf: str = ""  # character-stream accumulator; empty on field backbones

    def copy(self) -> "ConstraintState":
        s = ConstraintState(
            position=self.position,
            region=self.region,
            element_counts=dict(self.element_counts),
            composition_closed=self.composition_closed,
            lattice_params=dict(self.lattice_params),
            lattice_closed=self.lattice_closed,
            sites_emitted=self.sites_emitted,
        )
        s._buf = self._buf
        return s


class IncrementalParser:
    """Advances a ConstraintState one token at a time.

    Deliberately not a CIF validator. Its only job is to answer, at each step,
    which constraint quantities are determined by the prefix. A malformed stream
    leaves quantities undetermined rather than raising, because an untrained
    model emits malformed streams constantly and the mask emitter must still
    produce something well-defined.
    """

    def __init__(self, backbone: str):
        if backbone not in ("B1", "B1ref", "B2"):
            raise ValueError(
                f"backbone must be 'B1', 'B1ref', or 'B2', got {backbone!r}"
            )
        self.backbone = backbone

    def step(self, state: ConstraintState, token: str) -> ConstraintState:
        """Return the state after consuming one token. Does not mutate the input."""
        if self.backbone.startswith("B1"):
            if token == "<COMP_COMPLETE>":
                # Emitted by the reference ordering once the data_ tag's
                # composition has been consumed. Closing here is what makes
                # family (i) determined at the top of the file rather than in
                # the trailing atom-site loop. It closes only if a composition
                # was actually parsed, so a stream that never carried one cannot
                # claim a determined charge.
                s = state.copy()
                s.position += 1
                s.composition_closed = bool(s.element_counts)
                return s
            if len(token) == 1:
                return self._step_char(state, token)
        return self._step_field(state, token)

    def _step_char(self, state: ConstraintState, ch: str) -> ConstraintState:
        """Consume one CIF character.

        A character stream determines a lattice parameter only when its numeric
        literal TERMINATES — the prefix '_cell_length_a 5.12' does not determine
        `a`, because the next character may extend it to '5.123'. So the value is
        committed on the delimiter, not on the last digit. This one-token lag is
        not a rounding detail: it is the reason a character backbone can never
        make a numeric constraint active at the same position a field backbone
        can.
        """
        s = state.copy()
        s.position += 1
        s._buf = getattr(state, "_buf", "")  # type: ignore[attr-defined]

        if ch == "\n":
            # Flush on the line terminator only. A space cannot end a numeric
            # literal here because it separates the tag from its value on the
            # same line; flushing on space would discard the value before it is
            # read.
            buf = s._buf  # type: ignore[attr-defined]

            # A `data_` tag in the reference model's standardized ordering
            # carries the complete cell composition with counts, e.g.
            # `data_Na2Cl2`. When present it determines the composition at the
            # top of the file rather than in the trailing atom-site loop.
            if buf.startswith("data_") and len(buf) > 5:
                for el, cnt in re.findall(r"([A-Z][a-z]?)(\d+)", buf[5:]):
                    s.element_counts[el] = s.element_counts.get(el, 0) + int(cnt)

            # An atom-site line contributes one atom of its type symbol. This is
            # how a naive CIF layout (no composition in the data_ tag) accumulates
            # its composition: one line at a time, at the end of the file.
            elif s.region is Field.COMPOSITION:
                m = re.match(r"^([A-Z][a-z]?)\d*\s+([A-Z][a-z]?)\s", buf)
                if m:
                    el = m.group(2)
                    s.element_counts[el] = s.element_counts.get(el, 0) + 1
                    s.sites_emitted += 1
            # Real CIF spells angles `_cell_angle_alpha`, not
            # `_cell_length_alpha`. Matching only the length form parses lengths
            # and silently drops all three angles, leaving the lattice
            # permanently open — so the volume never becomes determined and
            # family (ii) reads as unusable. Both spellings are accepted:
            # the angle form because it is what CIF actually emits, the length
            # form because the naive fixture used it.
            for key, tags in (
                ("alpha", ("_cell_angle_alpha", "_cell_length_alpha")),
                ("beta", ("_cell_angle_beta", "_cell_length_beta")),
                ("gamma", ("_cell_angle_gamma", "_cell_length_gamma")),
                ("a", ("_cell_length_a",)),
                ("b", ("_cell_length_b",)),
                ("c", ("_cell_length_c",)),
            ):
                tag = next((tg for tg in tags if tg in buf), None)
                if tag is not None:
                    tail = buf.split(tag, 1)[1].strip()
                    try:
                        s.lattice_params[key] = float(tail)
                    except ValueError:
                        pass
                    break
            if len(s.lattice_params) >= 6:
                s.lattice_closed = True
            s._buf = ""  # type: ignore[attr-defined]
        else:
            s._buf = s._buf + ch  # type: ignore[attr-defined]
        return s

    def _step_field(self, state: ConstraintState, token: str) -> ConstraintState:
        s = state.copy()
        s.position += 1

        if token == "<COMP>":
            s.region = Field.COMPOSITION
        elif token == "<END_COMP>":
            s.composition_closed = True
            s.region = Field.LATTICE
        elif token == "<END_LATTICE>":
            # Only a block that actually carried all six parameters closes the
            # lattice. A marker alone must not mark the volume determined, or a
            # truncated stream would report a volume it never emitted.
            s.lattice_closed = len(s.lattice_params) >= 6
            s.region = Field.SITES
        elif token == "<EOS>":
            s.region = Field.DONE
        elif s.region is Field.COMPOSITION and token.startswith("el:"):
            _, el, cnt = token.split(":")
            s.element_counts[el] = s.element_counts.get(el, 0) + int(cnt)
        elif s.region is Field.LATTICE and token.startswith("lat:"):
            _, key, val = token.split(":")
            s.lattice_params[key] = float(val)
        elif s.region is Field.SITES and token.startswith("site:"):
            s.sites_emitted += 1

        return s

    def run(self, tokens: list[str]) -> Iterator[ConstraintState]:
        """Yield the state after each token, starting from the empty prefix."""
        s = ConstraintState()
        yield s
        for tok in tokens:
            s = self.step(s, tok)
            yield s


def regime_S_mask(state: ConstraintState, vocab: list[str], max_count: int) -> list[bool]:
    """Regime-S per-step mask: which element tokens keep the budget satisfiable.

    Prefix-closed by construction — a spent budget cannot be recovered — so this
    mask is exact rather than a heuristic, and no straight-through estimator is
    needed to train through it.
    """
    out = []
    for tok in vocab:
        if tok.startswith("el:"):
            _, el, cnt = tok.split(":")
            out.append(state.element_counts.get(el, 0) + int(cnt) <= max_count)
        else:
            out.append(True)
    return out
