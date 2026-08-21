"""The crystal mass budget must be dimensionally consistent with its channel."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.crystal.mp20 import ATOMIC_MASS, CrystalVocab, ELEMENTS, load_split
from scripts.train.train_crystal import (
    CrystalTrainConfig, GrammarMask, _element_position, _mass_budget, pad_batch,
)

from _devtree import requires_mp20

pytestmark = requires_mp20


@pytest.fixture(scope="module")
def data():
    cfg = CrystalTrainConfig(arm="projected")
    tr = load_split("train", limit=400)
    V = CrystalVocab(ELEMENTS, order=cfg.order)
    gm = GrammarMask(V, order=cfg.order)
    ids = pad_batch([V.encode(c) for c in tr], V.index["<EOS>"])[:64]
    mass = np.array([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                     for t in V.tokens])
    return cfg, tr, V, gm, ids, mass


def _demands(cfg, V, ids, mass):
    out = []
    for r in range(len(ids)):
        for t in range(1, ids.shape[1]):
            if not V.tokens[ids[r, t]].startswith("el:"):
                continue
            bud, slt = _mass_budget(ids[r:r + 1, :t], V, cfg)
            if np.isfinite(bud[0]):
                out.append((bud[0] / max(slt[0], 1), mass[ids[r, t]]))
    return np.array(out)


def test_demanded_mass_is_on_the_same_scale_as_the_data(data):
    """A per-atom channel needs a per-atom denominator, or the target is nonsense."""
    cfg, _, V, _, ids, mass = data
    g = _demands(cfg, V, ids, mass)
    ratio = np.median(g[:, 0]) / np.median(g[:, 1])
    assert 0.5 < ratio < 2.0, (
        f"constraint demands {ratio:.2f}x the mass the data's own next element "
        "carries; the budget denominator is not on the channel's footing"
    )


def test_few_demands_exceed_the_heaviest_available_element(data):
    """A target above every attainable value is infeasible by construction."""
    cfg, _, V, _, ids, mass = data
    g = _demands(cfg, V, ids, mass)
    frac = float((g[:, 0] > mass.max()).mean())
    assert frac < 0.15, (
        f"{frac:.1%} of actionable positions demand more mass than the heaviest "
        f"element ({mass.max():.0f}u); the budget is mis-scaled"
    )


def test_budget_is_undefined_until_the_cell_closes(data):
    """Density fixes a mass budget only once the volume is known."""
    cfg, _, V, _, ids, _ = data
    early, _ = _mass_budget(ids[:, :2], V, cfg)
    assert np.all(np.isnan(early)), (
        "a mass budget was reported before the cell block closed; the volume that "
        "a target density needs is not determined there"
    )
    late, _ = _mass_budget(ids[:, :14], V, cfg)
    assert np.any(np.isfinite(late)), "budget never becomes defined after the cell"


def test_actionable_positions_are_exactly_the_element_positions(data):
    """With a static channel only element positions can move the constraint."""
    cfg, _, V, gm, ids, mass = data
    M = gm.masks_for(ids)
    for r in range(8):
        for t in range(1, min(ids.shape[1], len(V.encode(load_split('train', limit=1)[0])) + 6)):
            is_el = _element_position(ids[r:r + 1, :t], V)[0]
            spread = mass[M[r, t]]
            if spread.max() > 0:
                assert is_el, (
                    f"position {t} carries nonzero channel mass but was not marked "
                    "actionable"
                )


def test_atom_count_denominator_tracks_the_data(data):
    """The configured atom estimate must match MP-20, not a guess."""
    cfg, tr, _, _, _, _ = data
    mean_atoms = float(np.mean([c.n_atoms for c in tr]))
    assert abs(cfg.expected_atoms - mean_atoms) < 2.0, (
        f"expected_atoms={cfg.expected_atoms} against a data mean of {mean_atoms:.1f}"
    )
