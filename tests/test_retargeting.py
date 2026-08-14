"""Tests for checkpoint persistence and zero-shot retargeting.

These exist because the retargeting claim is a claim about what a checkpoint carries.
If a checkpoint silently loses the target it trained at, or if the retargeting script
overwrites it with the deployment target, the matrix's diagonal becomes
indistinguishable from its off-diagonal and the headline claim becomes unfalsifiable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from scripts.crystal.retarget import FRONTIER_TARGETS, load_checkpoint, retarget  # noqa: E402
from scripts.train.train_crystal import (  # noqa: E402
    ELEMENTS,
    CrystalDecoder,
    CrystalTrainConfig,
    CrystalVocab,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _fake_checkpoint(tmp_path: Path, trained_at: float = 5.0, order: str = "cell_first"):
    """A checkpoint written the same way the trainer writes one, at tiny scale."""
    cfg = CrystalTrainConfig(arm="A3", d_model=32, n_layer=1, n_head=2,
                             density_target=trained_at, order=order)
    V = CrystalVocab(ELEMENTS, order=order)
    max_len = 26
    model = CrystalDecoder(len(V), max_len + 2, cfg)
    p = tmp_path / "ck.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "train_config": {**cfg.__dict__},
        "trained_density_target": trained_at,
        "max_len": max_len,
        "vocab_order": order,
        "vocab_size": len(V),
        "n_parameters": sum(q.numel() for q in model.parameters()),
        "elements": list(ELEMENTS),
        "schema_version": 1,
    }, p)
    return p, cfg, V, max_len


def test_checkpoint_records_the_target_it_trained_at(tmp_path):
    """The trained target must survive the round trip.

    Retargeting is only meaningful relative to where the checkpoint started. A
    checkpoint that does not carry its own training target cannot be placed on the
    matrix at all.
    """
    p, _, _, _ = _fake_checkpoint(tmp_path, trained_at=3.0)
    model, ck = load_checkpoint(p, torch.device("cpu"))
    assert ck["trained_density_target"] == 3.0
    assert ck["train_config"]["density_target"] == 3.0


def test_checkpoint_rebuild_is_exact(tmp_path):
    """Rebuilt weights must equal saved weights, parameter by parameter.

    A retargeting result read off a model that differs from the trained one measures
    the rebuild, not the transfer.
    """
    p, cfg, V, max_len = _fake_checkpoint(tmp_path)
    model, ck = load_checkpoint(p, torch.device("cpu"))
    saved = ck["state_dict"]
    live = model.state_dict()
    assert set(saved) == set(live)
    for k in saved:
        assert torch.equal(saved[k].cpu(), live[k].cpu()), f"parameter {k} changed on reload"


def test_rebuild_rejects_vocabulary_drift(tmp_path):
    """A checkpoint whose stored vocab size disagrees with the rebuilt one must raise.

    Silently padding or truncating would produce a model that samples from a different
    grammar than it trained on, and the failure would show up as a plausible-looking
    density rather than as an error.
    """
    p, _, _, _ = _fake_checkpoint(tmp_path)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ck["vocab_size"] = ck["vocab_size"] + 1
    torch.save(ck, p)
    with pytest.raises(ValueError, match="vocab size"):
        load_checkpoint(p, torch.device("cpu"))


def test_rebuild_rejects_architecture_drift(tmp_path):
    """A stored parameter count that disagrees with the rebuilt model must raise."""
    p, _, _, _ = _fake_checkpoint(tmp_path)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ck["n_parameters"] = int(ck["n_parameters"]) + 1
    torch.save(ck, p)
    with pytest.raises(ValueError, match="parameters"):
        load_checkpoint(p, torch.device("cpu"))


def test_trainer_exposes_target_and_seed_and_ckpt():
    """The retargeting matrix needs several checkpoints differing only in target/seed.

    If these are not arguments, building the matrix means editing a dataclass between
    runs, which is exactly how a run gets mislabelled.
    """
    import inspect

    from scripts.train.train_crystal import main
    params = inspect.signature(main).parameters
    for name in ("density_target", "seed", "ckpt"):
        assert name in params, f"train_crystal.main is missing the {name} argument"


def test_frontier_targets_match_the_published_frontier():
    """Retargeting columns must be the frontier's own targets, or the diagonal cannot
    be compared against the existing per-target result."""
    rows = RESULTS / "WS3_steering_frontier_rows.json"
    if not rows.exists():
        pytest.skip("frontier artifact absent")
    import json
    payload = json.loads(rows.read_text())["payload"]["rows"]
    published = sorted({v["target"] for v in payload.values()})
    assert published == sorted(FRONTIER_TARGETS), (
        f"retarget targets {sorted(FRONTIER_TARGETS)} disagree with the published "
        f"frontier {published}"
    )


def test_retargeting_artifact_keeps_train_and_deploy_separate():
    """Every emitted row must carry BOTH targets and an explicit diagonal flag.

    The claim is about off-diagonal cells. A row that records only one target cannot
    say whether it is evidence for the claim or the pre-existing result restated.
    """
    art = RESULTS / "WS4_zero_shot_retargeting.json"
    if not art.exists():
        pytest.skip("retargeting artifact absent; run scripts.crystal.retarget")
    import json
    payload = json.loads(art.read_text())["payload"]
    seen_off_diagonal = False
    for block in payload["matrix"]:
        for row in block["rows"]:
            for field in ("trained_density_target", "deploy_target", "is_diagonal"):
                assert field in row, f"row missing {field}"
            expected = abs(row["deploy_target"] - row["trained_density_target"]) < 1e-9
            assert row["is_diagonal"] is expected, (
                f"is_diagonal={row['is_diagonal']} disagrees with the targets "
                f"{row['trained_density_target']} -> {row['deploy_target']}"
            )
            if not row["is_diagonal"]:
                seen_off_diagonal = True
    assert seen_off_diagonal, "artifact contains no off-diagonal cell, so it tests nothing"


def test_retargeting_validity_is_exact_everywhere():
    """Regime S is a property of the mask, so it must hold at deployment targets the
    checkpoint never trained for. If it does not, the mask claim is narrower than
    stated and the scope section is wrong."""
    art = RESULTS / "WS4_zero_shot_retargeting.json"
    if not art.exists():
        pytest.skip("retargeting artifact absent")
    import json
    payload = json.loads(art.read_text())["payload"]
    for block in payload["matrix"]:
        for row in block["rows"]:
            assert row["validity"] == 1.0, (
                f"validity {row['validity']} at train {row['trained_density_target']} "
                f"-> deploy {row['deploy_target']}; the mask claim does not survive "
                f"retargeting"
            )
