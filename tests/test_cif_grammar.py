"""CIF grammar soundness, non-triviality, and the syntactic/semantic boundary.

Soundness is checked against pymatgen's CifParser, which shares no code with the
emitter. That independence is load-bearing: it caught a missing required column
(`_atom_site_label`) that reading the specification had not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crystal.cif_grammar import (
    GrammarState,
    allowed,
    full_vocabulary,
    render,
    step,
)
from scripts.crystal.parse_validity import (
    builds_structure,
    parses,
    sample_masked,
)


def _sample(seed: int) -> str:
    return sample_masked(np.random.default_rng(seed))[0]


@pytest.mark.parametrize("seed", range(12))
def test_every_masked_sample_is_syntactically_valid(seed):
    """Soundness: the mask must make syntactic invalidity impossible.

    This is the whole claim of Regime S on this backbone. A single failure here is
    a defect in the grammar, not a statistical shortfall.
    """
    ok, kind = parses(_sample(seed))
    assert ok, f"seed {seed}: masked sample failed to parse ({kind})"


def test_mask_is_not_a_single_forced_path():
    """Non-triviality: 'valid by construction' must not mean 'one output'.

    If the mask admitted one character at every step the comparison against
    unconstrained decoding would be vacuous, so the branching factor and the
    diversity of completed samples are both asserted.
    """
    rng = np.random.default_rng(3)
    s, widths = GrammarState(), []
    for _ in range(900):
        a = allowed(s)
        if not a:
            break
        widths.append(len(a))
        s = step(s, a[rng.integers(len(a))])
    assert max(widths) > 5, f"widest choice was {max(widths)}"
    assert np.mean(widths) > 1.5, f"mean branching {np.mean(widths):.2f} too narrow"
    assert len({_sample(i) for i in range(20)}) > 15, "samples are nearly identical"


def test_grammar_terminates_in_the_done_phase():
    """A completed sample must reach `done`, not run out of length."""
    rng = np.random.default_rng(7)
    s = GrammarState()
    for _ in range(900):
        a = allowed(s)
        if not a:
            break
        s = step(s, a[rng.integers(len(a))])
    assert s.phase == "done"
    assert allowed(s) == ""


def test_mask_is_strictly_narrower_than_the_vocabulary():
    """The mask must actually exclude characters, at every step of a full run.

    A mask equal to the vocabulary would be no mask at all. Checked at every
    position rather than once, so a grammar that stops constraining partway
    through cannot pass.
    """
    vocab = set(full_vocabulary())
    rng = np.random.default_rng(2)
    s = GrammarState()
    for _ in range(900):
        a = allowed(s)
        if not a:
            break
        assert set(a) < vocab, f"mask at phase {s.phase!r} excludes nothing"
        s = step(s, a[rng.integers(len(a))])
    assert s.phase == "done"


def test_required_label_column_is_present():
    """Regression: pymatgen raises KeyError without `_atom_site_label`.

    Found by running the emitter's output through the independent parser rather
    than by reading the CIF specification, which is why the soundness oracle is
    third-party.
    """
    text = _sample(0)
    assert "_atom_site_label" in text
    ok, kind = parses(text)
    assert ok, kind


def test_syntactic_and_semantic_validity_are_measured_separately():
    """The scope boundary, asserted rather than described.

    Masking guarantees syntax and says nothing about physical coherence. If these
    two rates were equal the distinction would be untestable and the earlier
    conflation — which reported the masked arm at 47% when every sample was
    syntactically valid — could recur unnoticed.
    """
    syn = sem = 0
    for seed in range(24):
        text = _sample(seed)
        if parses(text)[0]:
            syn += 1
            if builds_structure(text)[0]:
                sem += 1
    assert syn == 24, f"only {syn}/24 syntactically valid"
    assert sem < syn, (
        "no semantic failures observed: either the fixture got lucky or the two "
        "judgements have collapsed onto each other"
    )


def test_absurd_angles_are_a_semantic_not_syntactic_failure():
    """Pin the specific mechanism behind the syntactic/semantic gap.

    A cell with near-degenerate angles parses cleanly and then fails to yield a
    lattice. Substituting right angles must convert a semantic failure into a
    success without touching the syntax.
    """
    import re

    bad = next(t for t in (_sample(i) for i in range(30))
               if parses(t)[0] and not builds_structure(t)[0])
    fixed = re.sub(r"(_cell_angle_\w+) [\d.]+", r"\1 90.0000", bad)
    assert parses(bad)[0] and parses(fixed)[0], "both must be syntactically valid"
    assert builds_structure(fixed)[0], "right angles should yield a structure"
