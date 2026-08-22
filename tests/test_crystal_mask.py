"""The crystal grammar mask must accept the data and constrain generation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from _devtree import requires_mp20
from cpt.crystal.mp20 import ELEMENTS, CrystalVocab, load_split
from cpt.train.train_crystal import (
    CrystalDecoder,
    CrystalTrainConfig,
    GrammarMask,
    pad_batch,
    sample,
    structural_metrics,
)

pytestmark = requires_mp20


@pytest.fixture(scope="module")
def setup():
    tr = load_split("train", limit=300)
    V = CrystalVocab(ELEMENTS)
    return tr, V, GrammarMask(V)


def test_mask_accepts_every_real_transition(setup):
    """If the data's own sequences are illegal, the mask is wrong, not the data."""
    tr, V, gm = setup
    enc = [V.encode(c) for c in tr]
    ids = pad_batch(enc, V.index["<EOS>"])
    m = gm.masks_for(ids)
    bad = [(r, t) for r in range(len(enc)) for t in range(1, len(enc[r]))
           if not m[r, t, enc[r][t]]]
    assert not bad, f"{len(bad)} real transitions rejected, first at {bad[:1]}"


def test_mask_is_strictly_narrower_than_the_vocabulary(setup):
    """A mask that admits everything would make validity-by-construction vacuous."""
    tr, V, gm = setup
    ids = pad_batch([V.encode(c) for c in tr[:64]], V.index["<EOS>"])
    sizes = gm.masks_for(ids).sum(-1)
    assert sizes.max() < len(V), "mask never excludes anything"
    assert sizes.min() >= 1, "mask left a position with no legal token"


def test_masked_generation_is_structurally_valid_even_untrained(setup):
    """Validity comes from the mask, not from training. An untrained model proves it."""
    tr, V, gm = setup
    cfg = CrystalTrainConfig(arm="mask_only")
    ml = max(len(V.encode(c)) for c in tr)
    torch.manual_seed(0)
    model = CrystalDecoder(len(V), ml + 2, cfg)
    seqs, _ = sample(model, V, gm, cfg, torch.device("cpu"), 64, use_mask=True,
                     project_at_inference=False, max_len=ml + 2,
                     rng=np.random.default_rng(0))
    m = structural_metrics(seqs, V, gm, cfg)
    assert m["structural_validity_rate"] == 1.0, (
        f"masked generation gave {m['structural_validity_rate']:.4f}; the mask is "
        "supposed to make invalid structure unreachable"
    )


def test_sampler_does_not_duplicate_tokens(setup):
    """Pins the second off-by-one: an emitted token must not be re-emitted."""
    tr, V, gm = setup
    cfg = CrystalTrainConfig(arm="mask_only")
    ml = max(len(V.encode(c)) for c in tr)
    torch.manual_seed(0)
    model = CrystalDecoder(len(V), ml + 2, cfg)
    seqs, _ = sample(model, V, gm, cfg, torch.device("cpu"), 32, use_mask=True,
                     project_at_inference=False, max_len=ml + 2,
                     rng=np.random.default_rng(0))
    for s in seqs:
        toks = [V.tokens[i] for i in s]
        assert toks.count("<COMP>") == 1, f"<COMP> emitted {toks.count('<COMP>')} times"
        assert toks.count("<CELL>") == 1, f"<CELL> emitted {toks.count('<CELL>')} times"
        assert sum(t.startswith("len:") for t in toks) == 3
        assert sum(t.startswith("ang:") for t in toks) == 3


def test_unmasked_generation_is_not_valid_by_accident(setup):
    """The comparison arm must actually fail, or the unconstrained-vs-mask-only delta is empty."""
    tr, V, gm = setup
    cfg = CrystalTrainConfig(arm="unconstrained")
    ml = max(len(V.encode(c)) for c in tr)
    torch.manual_seed(0)
    model = CrystalDecoder(len(V), ml + 2, cfg)
    seqs, _ = sample(model, V, gm, cfg, torch.device("cpu"), 64, use_mask=False,
                     project_at_inference=False, max_len=ml + 2,
                     rng=np.random.default_rng(0))
    m = structural_metrics(seqs, V, gm, cfg)
    assert m["structural_validity_rate"] < 1.0, (
        "an untrained UNMASKED model produced only valid structures; the validity "
        "check is too weak to distinguish the arms"
    )


def test_density_decoding_round_trips_within_discretization(setup):
    """Decoded density must track the source structure, or the dial is meaningless."""
    tr, V, _ = setup
    err = [abs(V.decode_density(V.encode(c)) - c.density) / c.density
           for c in tr[:200] if V.decode_density(V.encode(c))]
    assert np.median(err) < 0.02, f"median relative error {np.median(err):.4f} too large"
