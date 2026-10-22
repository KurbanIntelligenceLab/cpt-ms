"""CIF grammar as a per-character mask emitter."""

from __future__ import annotations

from dataclasses import dataclass

DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"

ELEMENTS = ("H", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
            "K", "Ca", "Ti", "Fe", "Co", "Ni", "Cu", "Zn")

_CELL_TAGS = ("_cell_length_a", "_cell_length_b", "_cell_length_c",
              "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
_SYM_TAG = "_symmetry_space_group_name_H-M"
_LOOP_HEAD = ("loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")


@dataclass
class GrammarState:
    """Position in the CIF skeleton, as a function of the prefix only."""

    phase: str = "data_tag"
    idx: int = 0
    field_len: int = 0
    seen_dot: bool = False
    cell_i: int = 0
    head_i: int = 0
    site_col: int = 0
    n_sites: int = 0
    elem_prefix: str = ""

    def copy(self) -> GrammarState:
        return GrammarState(**self.__dict__)


MIN_SITES = 1
MAX_SITES = 4
FORMULA_MIN, FORMULA_MAX = 2, 8
INT_DIGITS, FRAC_DIGITS = 1, 4


def _num_allowed(s: GrammarState, *, allow_terminator: str) -> str:
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
        out = ""
        cands = [e for e in ELEMENTS if e.startswith(s.elem_prefix)]
        for e in cands:
            if len(e) > len(s.elem_prefix):
                out += e[len(s.elem_prefix)]
        if s.elem_prefix in ELEMENTS:
            out += " "
        return "".join(dict.fromkeys(out))

    if p == "site_label":
        if s.field_len == 0:
            return DIGITS
        return DIGITS + " " if s.field_len < 2 else " "

    if p == "site_number":
        return _num_allowed(s, allow_terminator=" " if s.site_col < 3 else "\n")

    if p == "row_boundary":
        out = ""
        if s.n_sites < MAX_SITES:
            out += DIGITS
        if s.n_sites >= MIN_SITES:
            out += "\x03"
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
