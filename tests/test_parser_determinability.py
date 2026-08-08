"""Verification for the incremental parser and the determinability measurement.

Each test checks a property by a route independent of the implementation: the
volume against its closed-form value, prefix-purity against recomputation from
scratch, and determinability positions against brute-force scanning.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crystal.determinability import build_b1, build_b2
from scripts.crystal.family_ii import family_i_profile, family_ii_profile, volume_determined_at
from scripts.crystal.parser import ConstraintState, IncrementalParser, regime_S_mask


def test_cubic_volume_is_exact():
    """A 5 A cube must give exactly 125 cubic angstroms."""
    st = list(IncrementalParser("B2").run(build_b2(2, 4)))[-1]
    assert st.cell_volume == pytest.approx(125.0, abs=1e-9)


def test_triclinic_volume_matches_closed_form():
    """The general formula against an independently computed triclinic cell."""
    p = IncrementalParser("B2")
    toks = ["<COMP>", "el:A:1", "<END_COMP>",
            "lat:a:4.0", "lat:b:5.0", "lat:c:6.0",
            "lat:alpha:80.0", "lat:beta:95.0", "lat:gamma:110.0", "<END_LATTICE>"]
    st = list(p.run(toks))[-1]
    al, be, ga = math.radians(80.0), math.radians(95.0), math.radians(110.0)
    expect = 4.0*5.0*6.0*math.sqrt(
        1 - math.cos(al)**2 - math.cos(be)**2 - math.cos(ga)**2
        + 2*math.cos(al)*math.cos(be)*math.cos(ga))
    assert st.cell_volume == pytest.approx(expect, rel=1e-12)


def test_volume_is_none_before_lattice_closes():
    """No volume may be reported from an incomplete lattice block."""
    p = IncrementalParser("B2")
    toks = ["<COMP>", "el:A:1", "<END_COMP>", "lat:a:4.0", "lat:b:5.0"]
    for st in p.run(toks):
        assert st.cell_volume is None


def test_end_lattice_marker_alone_does_not_determine_volume():
    """A closing marker without six parameters must not claim determination."""
    p = IncrementalParser("B2")
    toks = ["<COMP>", "el:A:1", "<END_COMP>", "lat:a:4.0", "<END_LATTICE>"]
    st = list(p.run(toks))[-1]
    assert st.lattice_closed is False
    assert st.cell_volume is None


@pytest.mark.parametrize("backbone,builder", [("B1", build_b1), ("B2", build_b2)])
def test_state_is_a_pure_function_of_the_prefix(backbone, builder):
    """Stepping to position k equals re-running the first k tokens from scratch.

    This is the property the projection layer depends on: no state may leak
    information from tokens the model has not emitted yet.
    """
    toks = builder(3, 8)
    p = IncrementalParser(backbone)
    states = list(p.run(toks))
    for k in (1, len(toks)//3, len(toks)//2, len(toks)-1):
        fresh = list(p.run(toks[:k]))[-1]
        assert fresh.element_counts == states[k].element_counts
        assert fresh.lattice_params == states[k].lattice_params
        assert fresh.lattice_closed == states[k].lattice_closed
        assert fresh.cell_volume == states[k].cell_volume


@pytest.mark.parametrize("backbone,builder", [("B1", build_b1), ("B2", build_b2)])
def test_determinability_position_is_the_first_such_position(backbone, builder):
    """Brute force: no earlier prefix determines the volume."""
    toks = builder(3, 8)
    pos = volume_determined_at(toks, backbone)
    assert pos is not None
    p = IncrementalParser(backbone)
    assert list(p.run(toks[:pos]))[-1].cell_volume is not None
    assert list(p.run(toks[:pos-1]))[-1].cell_volume is None


def test_b1_charge_is_near_terminal_and_b2_is_not():
    """The backbone asymmetry the report turns on, asserted as a property.

    On a CIF character stream the composition is spelled out in the atom-site
    loop at the end, so charge balance is determined only in the final few
    percent of the sequence. On a structured stream it closes early.
    """
    t1, t2 = build_b1(3, 8), build_b2(3, 8)
    assert family_i_profile(t1, "B1").usable_fraction < 0.05
    assert family_i_profile(t2, "B2").usable_fraction > 0.5


def test_reference_ordering_makes_charge_determined_early():
    """The B1 blocker dissolves under the reference model's own field order.

    CrystaLLM emits `data_{comp_str}` as the first line, carrying the complete
    cell composition with counts. Parsing it there determines charge balance at
    the top of the file instead of in the trailing atom-site loop, which is the
    difference between a constraint that can carry a steering sweep and one that
    cannot.
    """
    from scripts.crystal.determinability import build_b1_reference

    for ne, ns in [(2, 4), (3, 8), (5, 20)]:
        naive = family_i_profile(build_b1(ne, ns), "B1")
        ref = family_i_profile(build_b1_reference(ne, ns), "B1ref")
        assert naive.usable_fraction < 0.01, f"{ne}el_{ns}site naive"
        assert ref.usable_fraction > 0.9, f"{ne}el_{ns}site reference"


def test_reference_ordering_keeps_density_usable():
    """Moving composition first must not cost family (ii).

    Regression for a real defect: the reference fixture used CIF's actual
    `_cell_angle_*` spelling while the parser matched only `_cell_length_*`, so
    all three angles were dropped, the lattice never closed, and density read as
    unusable everywhere.
    """
    from scripts.crystal.determinability import build_b1_reference

    for ne, ns in [(2, 4), (3, 8), (5, 20)]:
        toks = build_b1_reference(ne, ns)
        st = list(IncrementalParser("B1ref").run(toks))[-1]
        assert st.cell_volume == pytest.approx(125.0, abs=1e-9)
        assert family_ii_profile(toks, "B1ref").usable_fraction > 0.3


def test_cif_angle_tags_are_parsed():
    """`_cell_angle_alpha` is what real CIF writes; it must be understood.

    Matching only `_cell_length_alpha` parses lengths and silently drops angles,
    leaving the lattice permanently open. That fails on any actual CIF file.
    """
    p = IncrementalParser("B1")
    cif = ("_cell_length_a 4.0000\n_cell_length_b 5.0000\n_cell_length_c 6.0000\n"
           "_cell_angle_alpha 90.0000\n_cell_angle_beta 90.0000\n_cell_angle_gamma 90.0000\n")
    st = list(p.run(list(cif)))[-1]
    assert set(st.lattice_params) == {"a", "b", "c", "alpha", "beta", "gamma"}
    assert st.cell_volume == pytest.approx(120.0, abs=1e-9)


def test_composition_marker_requires_a_parsed_composition():
    """A marker alone must not claim a determined charge."""
    p = IncrementalParser("B1ref")
    st = list(p.run(list("data_\n") + ["<COMP_COMPLETE>"]))[-1]
    assert st.composition_closed is False
    assert st.n_atoms is None


def test_verdicts_are_derived_not_asserted():
    """No narrative field in the artifact may contradict the data beside it.

    A hand-written verdict stored next to its own computation can drift from it
    silently when the computation changes. This test regenerates the payload and
    checks every verdict band follows from its own summary figure, so a future
    change to the parser cannot leave a stale claim in the artifact.
    """
    from scripts.crystal.determinability import main
    import json

    path = main(smoke=True)
    payload = json.loads(Path(path).read_text())["payload"]
    for key, v in payload["verdict"].items():
        mean = v["mean_usable_fraction"]
        if v["n_cases_never_determined"]:
            expect = "never determined"
        elif mean < 0.05:
            expect = "near-terminal"
        elif mean < 0.35:
            expect = "late"
        else:
            expect = "usable over most of the sequence"
        assert v["band"] == expect, f"{key}: band {v['band']!r} vs mean {mean}"
        assert f"{mean:.3f}" in v["statement"]


def test_regime_S_mask_is_prefix_closed():
    """Once a budget is spent it cannot reopen: masks only shrink."""
    vocab = ["el:A:1", "el:B:1", "<EOS>"]
    p = IncrementalParser("B2")
    s = ConstraintState(region=__import__("scripts.crystal.parser", fromlist=["Field"]).Field.COMPOSITION)
    prev = regime_S_mask(s, vocab, max_count=2)
    for _ in range(3):
        s = p.step(s, "el:A:1")
        cur = regime_S_mask(s, vocab, max_count=2)
        assert all(c <= q for c, q in zip(cur, prev))
        prev = cur
    assert prev[0] is False
