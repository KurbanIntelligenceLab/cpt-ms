"""Incremental constraint-state parser and mask emitter."""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum


class Field(Enum):
    """CIF-ish field regions, in the order a structured stream emits them."""

    PRELUDE = "prelude"
    COMPOSITION = "composition"
    LATTICE = "lattice"
    SITES = "sites"
    DONE = "done"


@dataclass
class ConstraintState:
    """Everything the projection layer may condition on at one decoding step."""

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
        """Cell volume in cubic angstroms, or None while the lattice is open."""
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

    _buf: str = ""

    def copy(self) -> ConstraintState:
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
    """Advances a ConstraintState one token at a time."""

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
                s = state.copy()
                s.position += 1
                s.composition_closed = bool(s.element_counts)
                return s
            if len(token) == 1:
                return self._step_char(state, token)
        return self._step_field(state, token)

    def _step_char(self, state: ConstraintState, ch: str) -> ConstraintState:
        s = state.copy()
        s.position += 1
        s._buf = getattr(state, "_buf", "")  # type: ignore[attr-defined]

        if ch == "\n":
            buf = s._buf  # type: ignore[attr-defined]

            if buf.startswith("data_") and len(buf) > 5:
                for el, cnt in re.findall(r"([A-Z][a-z]?)(\d+)", buf[5:]):
                    s.element_counts[el] = s.element_counts.get(el, 0) + int(cnt)

            elif s.region is Field.COMPOSITION:
                m = re.match(r"^([A-Z][a-z]?)\d*\s+([A-Z][a-z]?)\s", buf)
                if m:
                    el = m.group(2)
                    s.element_counts[el] = s.element_counts.get(el, 0) + 1
                    s.sites_emitted += 1
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
                    with contextlib.suppress(ValueError):
                        s.lattice_params[key] = float(tail)
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
    """Regime-S per-step mask: which element tokens keep the budget satisfiable."""
    out = []
    for tok in vocab:
        if tok.startswith("el:"):
            _, el, cnt = tok.split(":")
            out.append(state.element_counts.get(el, 0) + int(cnt) <= max_count)
        else:
            out.append(True)
    return out
