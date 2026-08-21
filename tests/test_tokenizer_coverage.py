"""Discretization ranges must COVER the data, not merely span its typical values."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.crystal.mp20 import CrystalVocab, ELEMENTS, load_split

from _devtree import requires_mp20

pytestmark = requires_mp20


@pytest.fixture(scope="module")
def data():
    return load_split("train"), CrystalVocab(ELEMENTS, order="cell_first")


def test_no_angle_is_clamped(data):
    tr, V = data
    angs = np.array([a for c in tr for a in c.angles])
    outside = int(((angs < V.ANG_LO) | (angs > V.ANG_HI)).sum())
    assert outside == 0, (
        f"{outside} angles fall outside [{V.ANG_LO}, {V.ANG_HI}]; data range is "
        f"[{angs.min():.2f}, {angs.max():.2f}]"
    )


def test_no_length_is_clamped(data):
    tr, V = data
    lens = np.array([L for c in tr for L in c.lengths])
    outside = int(((lens < V.LEN_LO) | (lens > V.LEN_HI)).sum())
    assert outside == 0, (
        f"{outside} lengths fall outside [{V.LEN_LO}, {V.LEN_HI}]; data range is "
        f"[{lens.min():.2f}, {lens.max():.2f}]"
    )


def test_decode_error_is_bounded_in_the_high_density_tail(data):
    """Aggregate error hides tail damage, so the tail is checked on its own."""
    tr, V = data
    hi = [c for c in tr if c.density > 19.5]
    assert len(hi) >= 10, "too few high-density structures for this check to bind"
    err = np.array([abs(V.decode_density(V.encode(c)) - c.density) / c.density
                    for c in hi])
    assert err.max() < 0.10, (
        f"worst high-density decode error {err.max():.3f}; the tail targets would be "
        "measuring discretization rather than steering"
    )


def test_bulk_accuracy_is_not_traded_away_for_tail_accuracy(data):
    """Widening the range spreads the same buckets over more span; check the cost."""
    tr, V = data
    bulk = [c for c in tr[:3000] if 4 < c.density < 9]
    err = np.array([abs(V.decode_density(V.encode(c)) - c.density) / c.density
                    for c in bulk])
    assert np.median(err) < 0.02, f"bulk median error {np.median(err):.4f}"
