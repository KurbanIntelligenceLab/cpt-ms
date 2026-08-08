"""CIF grammar as a per-character mask emitter (WS-2).

Regime S for the B1 backbone. At every decoding step the emitter returns the set
of characters that keep the prefix extendable to a syntactically valid CIF. A
decoder restricted to that set cannot emit an unparseable file — the syntactic
failure mode is removed by construction rather than reduced.

The grammar is a deterministic finite-state machine over a skeleton CIF:

    data_<formula>
    _cell_length_a <num>      ... b, c
    _cell_angle_alpha <num>   ... beta, gamma
    _symmetry_space_group_name_H-M P 1
    loop_
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    <El> <num> <num> <num>    (one or more)

Two properties matter and both are testable rather than asserted:

  SOUNDNESS   every string the emitter allows to completion parses under an
              independent third-party parser. Tested against pymatgen's
              CifParser, which shares no code with this module. If a masked
              sample fails to parse, the grammar is incomplete and that is a
              finding about this emitter, not about the method.

  NON-TRIVIALITY  the mask must not collapse to a single legal character at
              every step, or "valid by construction" would mean "one possible
              output" and the comparison against unconstrained decoding would be
              vacuous. Branching factor is measured and reported.

The skeleton is deliberately narrow. It covers the subset a generative model must
emit to be evaluable, not the full CIF dictionary, and the reported claim is
scoped to that subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"

# Elements the grammar admits. Two-letter symbols make the element field a real
# branching decision rather than a single character.
ELEMENTS = ("H", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
            "K", "Ca", "Ti", "Fe", "Co", "Ni", "Cu", "Zn")

_CELL_TAGS = ("_cell_length_a", "_cell_length_b", "_cell_length_c",
              "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
_SYM_TAG = "_symmetry_space_group_name_H-M"
# `_atom_site_label` is not decoration: pymatgen's CifParser raises KeyError
# without it. Discovered by running the emitter's own output through the
# independent parser rather than by reading the CIF specification, which is the
# reason the soundness check uses a third-party oracle.
_LOOP_HEAD = ("loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")


@dataclass
class GrammarState:
    """Position in the CIF skeleton, as a function of the prefix only."""

    phase: str = "data_tag"        # which literal or field is being emitted
    idx: int = 0                   # offset within the current literal
    field_len: int = 0             # characters emitted in the current free field
    seen_dot: bool = False         # numeric field has consumed its decimal point
    cell_i: int = 0                # which of the six cell parameters
    head_i: int = 0                # which loop-header line
    site_col: int = 0             # column within an atom-site row
    n_sites: int = 0               # completed atom-site rows
    elem_prefix: str = ""          # partial element symbol

    def copy(self) -> "GrammarState":
        return GrammarState(**self.__dict__)


MIN_SITES = 1
MAX_SITES = 4
FORMULA_MIN, FORMULA_MAX = 2, 8
INT_DIGITS, FRAC_DIGITS = 1, 4


def _num_allowed(s: GrammarState, *, allow_terminator: str) -> str:
    """Characters legal inside a fixed-format number `d.dddd`.

    Numbers are emitted as one integer digit, a point, then exactly four
    fractional digits. Fixing the format keeps the FSM deterministic without
    making the field a single choice: each digit position is a free decision.
    """
    if not s.seen_dot:
        if s.field_len == 0:
            return DIGITS
        return "."
    if s.field_len <= FRAC_DIGITS:
        return DIGITS
    return allow_terminator


def allowed(state: GrammarState) -> str:
    """Characters that keep the prefix completable to a valid CIF."""
    s = state
    p = s.phase

    if p == "data_tag":
        lit = "data_"
        return lit[s.idx]

    if p == "formula":
        out = ""
        if s.field_len < FORMULA_MAX:
            out += UPPER + DIGITS if s.field_len else UPPER
        if s.field_len >= FORMULA_MIN:
            out += "\n"
        return out

    if p == "cell_tag":
        lit = _CELL_TAGS[s.cell_i]
        return lit[s.idx] if s.idx < len(lit) else " "

    if p == "cell_value":
        return _num_allowed(s, allow_terminator="\n")

    if p == "sym_tag":
        lit = _SYM_TAG
        return lit[s.idx] if s.idx < len(lit) else " "

    if p == "sym_value":
        lit = "P 1\n"
        return lit[s.idx]

    if p == "loop_head":
        lit = _LOOP_HEAD[s.head_i]
        return lit[s.idx] if s.idx < len(lit) else "\n"

    if p == "site_element":
        # Either continue a two-letter symbol or terminate a complete one.
        out = ""
        cands = [e for e in ELEMENTS if e.startswith(s.elem_prefix)]
        for e in cands:
            if len(e) > len(s.elem_prefix):
                out += e[len(s.elem_prefix)]
        if s.elem_prefix in ELEMENTS:
            out += " "
        return "".join(dict.fromkeys(out))

    if p == "site_label":
        # A per-row label. Digits only, so it is unambiguous and cannot collide
        # with the element column that follows.
        if s.field_len == 0:
            return DIGITS
        return DIGITS + " " if s.field_len < 2 else " "

    if p == "site_number":
        # Three coordinate columns; the third ends the row.
        return _num_allowed(s, allow_terminator=" " if s.site_col < 3 else "\n")

    if p == "row_boundary":
        out = ""
        if s.n_sites < MAX_SITES:
            out += DIGITS
        if s.n_sites >= MIN_SITES:
            out += "\x03"          # end-of-file marker
        return out

    if p == "done":
        return ""

    raise ValueError(f"unknown phase {p!r}")


def step(state: GrammarState, ch: str) -> GrammarState:
    """Advance the grammar by one character. Caller must pass a legal character."""
    s = state.copy()
    p = s.phase

    if p == "data_tag":
        s.idx += 1
        if s.idx == len("data_"):
            s.phase, s.idx, s.field_len = "formula", 0, 0
        return s

    if p == "formula":
        if ch == "\n":
            s.phase, s.idx, s.cell_i = "cell_tag", 0, 0
        else:
            s.field_len += 1
        return s

    if p == "cell_tag":
        lit = _CELL_TAGS[s.cell_i]
        if s.idx < len(lit):
            s.idx += 1
        else:
            s.phase, s.field_len, s.seen_dot = "cell_value", 0, False
        return s

    if p == "cell_value":
        if ch == "\n":
            s.cell_i += 1
            if s.cell_i < len(_CELL_TAGS):
                s.phase, s.idx = "cell_tag", 0
            else:
                s.phase, s.idx = "sym_tag", 0
        elif ch == ".":
            s.seen_dot, s.field_len = True, 0
        else:
            s.field_len += 1
        return s

    if p == "sym_tag":
        if s.idx < len(_SYM_TAG):
            s.idx += 1
        else:
            s.phase, s.idx = "sym_value", 0
        return s

    if p == "sym_value":
        s.idx += 1
        if s.idx == len("P 1\n"):
            s.phase, s.idx, s.head_i = "loop_head", 0, 0
        return s

    if p == "loop_head":
        lit = _LOOP_HEAD[s.head_i]
        if s.idx < len(lit):
            s.idx += 1
        else:
            s.head_i += 1
            s.idx = 0
            if s.head_i == len(_LOOP_HEAD):
                s.phase, s.field_len = "site_label", 0
        return s

    if p == "site_label":
        if ch == " ":
            s.phase, s.elem_prefix = "site_element", ""
        else:
            s.field_len += 1
        return s

    if p == "site_element":
        if ch == " ":
            s.phase, s.site_col = "site_number", 1
            s.field_len, s.seen_dot = 0, False
        else:
            s.elem_prefix += ch
        return s

    if p == "site_number":
        if ch in (" ", "\n"):
            if ch == "\n":
                s.n_sites += 1
                s.phase = "row_boundary"
            else:
                s.site_col += 1
                s.field_len, s.seen_dot = 0, False
        elif ch == ".":
            s.seen_dot, s.field_len = True, 0
        else:
            s.field_len += 1
        return s

    if p == "row_boundary":
        if ch == "\x03":
            s.phase = "done"
        else:
            s.phase, s.field_len = "site_label", 1
        return s

    raise ValueError(f"cannot step from phase {p!r}")


def full_vocabulary() -> str:
    """Every character the grammar can ever emit, plus the end marker."""
    chars = set("data_ \n.PH-M")
    chars |= set(DIGITS) | set(UPPER)
    for lit in (*_CELL_TAGS, _SYM_TAG, *_LOOP_HEAD):
        chars |= set(lit)
    for e in ELEMENTS:
        chars |= set(e)
    chars.add("\x03")
    return "".join(sorted(chars))


def render(tokens: list[str]) -> str:
    """Characters to CIF text, dropping the end marker."""
    return "".join(t for t in tokens if t != "\x03")
