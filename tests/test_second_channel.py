"""Tests for the second constraint family on real crystals.

Two claims are pinned here. First, that generalizing the mass channel over its weight
table did not change the density code path -- every published density artifact depends on
it. Second, that charge neutrality is genuinely outside the expressible class rather than
merely unimplemented, and that the second family chosen instead is not a restatement of
density.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("smact")

from scripts.crystal.second_channel import (  # noqa: E402
    _rank_ratio,
    charge_linearity_evidence,
    joint_conditioning,
    valence_table,
)
from scripts.crystal.mp20 import load_split  # noqa: E402
from scripts.train.train_crystal import (  # noqa: E402
    ATOMIC_MASS,
    ELEMENTS,
    CrystalVocab,
    GrammarMask,
    mass_channel,
    weight_channel,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _prefix_ids(n: int = 24):
    V = CrystalVocab(ELEMENTS, order="cell_first")
    train = load_split("train", limit=n)
    enc = [V.encode(c) for c in train]
    L = max(len(e) for e in enc)
    ids = np.full((len(enc), L), V.index["<EOS>"], dtype=np.int64)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = e
    return V, ids


def test_generalized_channel_reproduces_mass_channel_exactly():
    """weight_channel(ATOMIC_MASS) must equal mass_channel bit for bit.

    Every published density result runs through mass_channel. If the generalization
    changed its output by any amount, those artifacts would no longer describe the code
    that produced them.
    """
    V, ids = _prefix_ids()
    for cut in (3, 7, 11, 15, ids.shape[1]):
        a = mass_channel(ids[:, :cut], V)
        b = weight_channel(ids[:, :cut], V, ATOMIC_MASS)
        assert a.shape == b.shape
        assert np.array_equal(a, b), f"channels diverge at prefix length {cut}"


def test_channels_are_nonzero_on_the_next_token_support():
    """A channel that is identically zero on the allowed support is a vacuous constraint.

    The mask returned by masks_for at index -1 belongs to the token already emitted, so
    reading it directly yields a support on which the mass channel is all zeros -- an
    attainable range of {0} that reads as ill-conditioning rather than as a wrong
    position. This test pins the padded-mask convention that avoids it.
    """
    V, ids = _prefix_ids(8)
    gm = GrammarMask(V, order="cell_first")
    found_nonzero = False
    for r in range(ids.shape[0]):
        for cut in range(3, 16):
            pre = ids[r:r + 1, :cut]
            padded = np.concatenate(
                [pre, np.zeros((1, 1), dtype=pre.dtype)], axis=1)
            mk = gm.masks_for(padded)[0, -1, :]
            if mk.sum() <= 1:
                continue
            ch = weight_channel(pre, V, ATOMIC_MASS)[0][mk]
            if np.any(ch != 0):
                found_nonzero = True
                break
        if found_nonzero:
            break
    assert found_nonzero, (
        "no prefix produced a nonzero mass channel on its next-token support; the mask "
        "alignment is off by one position"
    )


def test_count_positions_are_exactly_collinear():
    """At a count position the two channels are one constraint written twice.

    With pending element E the tokens are cnt:k, so mass contributes MASS[E]*k and
    valence VALENCE[E]*k -- both proportional to k. Joint steering is impossible there
    for a structural reason, and the code must not treat such a position as a rank-2
    test case.
    """
    V, ids = _prefix_ids(12)
    gm = GrammarMask(V, order="cell_first")
    VE = {k: float(v) for k, v in valence_table().items()}
    seen = 0
    for r in range(ids.shape[0]):
        for cut in range(3, 20):
            pre = ids[r:r + 1, :cut]
            padded = np.concatenate([pre, np.zeros((1, 1), dtype=pre.dtype)], axis=1)
            mk = gm.masks_for(padded)[0, -1, :]
            if mk.sum() <= 1:
                continue
            toks = [V.tokens[i] for i in np.where(mk)[0]]
            if not all(t.startswith("cnt:") for t in toks):
                continue
            cm = weight_channel(pre, V, ATOMIC_MASS)[0][mk]
            cv = weight_channel(pre, V, VE)[0][mk]
            if np.unique(cm).size < 2 or np.unique(cv).size < 2:
                continue
            seen += 1
            assert _rank_ratio(cm, cv) < 1e-6, (
                f"count position has rank ratio {_rank_ratio(cm, cv):.3e}; expected the "
                f"two channels to be collinear there"
            )
    assert seen > 0, "no count position with channel variation was found to test"


def test_element_positions_are_rank_two():
    """Element positions must carry two genuinely independent constraints."""
    V, ids = _prefix_ids(12)
    gm = GrammarMask(V, order="cell_first")
    VE = {k: float(v) for k, v in valence_table().items()}
    ratios = []
    for r in range(ids.shape[0]):
        for cut in range(3, 20):
            pre = ids[r:r + 1, :cut]
            padded = np.concatenate([pre, np.zeros((1, 1), dtype=pre.dtype)], axis=1)
            mk = gm.masks_for(padded)[0, -1, :]
            if mk.sum() <= 1:
                continue
            toks = [V.tokens[i] for i in np.where(mk)[0]]
            if all(t.startswith("cnt:") for t in toks):
                continue
            cm = weight_channel(pre, V, ATOMIC_MASS)[0][mk]
            cv = weight_channel(pre, V, VE)[0][mk]
            if np.unique(cm).size < 2 or np.unique(cv).size < 2:
                continue
            ratios.append(_rank_ratio(cm, cv))
    assert ratios, "no element position with channel variation was found"
    assert min(ratios) > 1e-6, (
        f"an element position was collinear (min rank ratio {min(ratios):.3e}); the two "
        f"quantities would not be jointly steerable anywhere"
    )


def test_joint_solve_on_real_prefixes_is_exact_and_conditioned():
    """The crystal-data counterpart of the synthetic multi-constraint check."""
    out = joint_conditioning(n_prefixes=16, seed=0)
    assert out["all_converged"], f"{out['n_converged']}/{out['n_prefixes_with_choice']}"
    assert out["max_abs_residual"] < 1e-9, out["max_abs_residual"]
    assert out["n_ill_conditioned"] == 0, out["n_ill_conditioned"]
    assert out["mass_outside_grammar_mask"] == 0.0, (
        "the joint projection put mass on grammatically illegal tokens"
    )


def test_charge_neutrality_is_not_linear():
    """The scope claim must rest on a measurement, not an assertion.

    If a linear proxy were adequate the paper should use it; the measurement is what
    licenses excluding charge neutrality from the expressible class.
    """
    ev = charge_linearity_evidence(n_structures=300)
    assert ev["n_multivalent_elements"] > 0.5 * ev["n_elements_with_states"], (
        "most elements should be multivalent, which is what makes the screen existential"
    )
    assert ev["recall_of_smact_balanced"] < 0.5, (
        f"the linear proxy recovers {ev['recall_of_smact_balanced']:.3f} of the "
        f"SMACT-balanced set; if it were adequate the exclusion would be wrong"
    )
    assert ev["agreement"] < 0.9


def test_second_family_is_not_a_restatement_of_density():
    """Two nearly collinear constraints make the joint solve ill-posed by construction.

    The pair has to be correlated enough to be a real test of composition and far enough
    from collinear to stay conditioned, so the correlation is asserted inside a band
    rather than merely reported.
    """
    art = RESULTS / "second_channel.json"
    if not art.exists():
        pytest.skip("second-channel artifact absent; run scripts.crystal.second_channel")
    p = json.loads(art.read_text())["payload"]
    corr = p["channel_statistics"]["correlation"]
    assert 0.3 < abs(corr) < 0.95, (
        f"correlation {corr:.3f} between mass density and valence electron density is "
        f"outside the usable band: too low is an unrealistically easy pair, too high is "
        f"one constraint written twice"
    )
    assert p["second_family"]["directly_emitted"] is False


def test_valence_table_covers_the_split():
    """A channel undefined for some elements would silently zero their contribution."""
    train = load_split("train", limit=3000)
    els = {e for c in train for e in c.formula}
    VE = valence_table(sorted(els))
    missing = sorted(els - set(VE))
    assert not missing, f"no valence electron count for {missing}"
