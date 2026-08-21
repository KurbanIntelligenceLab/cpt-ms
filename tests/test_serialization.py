"""The expressible class is coupled to the token ordering, and that must stay measured.

This result replaced a requested experiment with an argument for why the experiment cannot be
run, which is exactly the shape of claim a reader should distrust unless it is measured. So
the measurement is recomputed here from the vocabulary and the trainer's own budget function
rather than read from the artifact, and the prose is checked against it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "results" / "serialization_expressibility.json"


def _p() -> dict:
    if not ART.exists():
        pytest.skip("serialization artifact absent")
    return json.loads(ART.read_text())["payload"]


def test_budget_determinability_recomputes_from_the_code():
    """Recompute on a small sample using the trainer's own budget function.

    If a future change made the budget determinable under composition-first, this test would
    fail and the paper's claim would have to change with it -- which is the point.
    """
    from scripts.crystal.mp20 import ATOMIC_MASS, ELEMENTS, CrystalVocab, load_split
    from scripts.train.train_crystal import CrystalTrainConfig, _additive_budget

    train = load_split("train", limit=60)
    cfg = CrystalTrainConfig(arm="A3", density_target=5.0)
    frac = {}
    for order in ("cell_first", "comp_first"):
        V = CrystalVocab(ELEMENTS, order=order)
        det = tot = 0
        for c in train:
            e = V.encode(c)
            ids = np.array([e], dtype=np.int64)
            for cut in range(2, len(e)):
                if not V.tokens[e[cut]].startswith("el:"):
                    continue
                b, slots = _additive_budget(ids[:, :cut], V, cfg, ATOMIC_MASS,
                                            cfg.density_target, 1.6605390666)
                tot += 1
                det += bool(np.isfinite(b[0]) and slots[0] > 0)
        frac[order] = det / tot if tot else 0.0

    assert frac["cell_first"] > 0.9, (
        f"the density budget is determined at only {frac['cell_first']:.1%} of element "
        f"positions under cell-first; the published steering results assume it is determined"
    )
    assert frac["comp_first"] < 0.05, (
        f"the budget IS determined at {frac['comp_first']:.1%} under composition-first, so "
        f"the claim that the second-serialization experiment cannot be run no longer holds "
        f"and the paper must be corrected"
    )


def test_artifact_agrees_with_the_recomputation():
    p = _p()
    rows = {r["order"]: r for r in p["rows"]}
    assert rows["cell_first"]["determined_fraction"] > 0.9
    assert rows["comp_first"]["determined_fraction"] < 0.05


def test_the_second_family_inherits_the_dependence():
    """Both constraint families are per-volume densities, so the coupling must be identical.

    Asserting this separately matters: if the two diverged, the mechanism explanation (the
    volume closing) would be wrong even if the density numbers looked right.
    """
    p = _p()
    for r in p["rows"]:
        assert abs(r["determined_fraction"] - r["valence_determined_fraction"]) < 1e-9, (
            f"under {r['order']} the two families differ in determinability "
            f"({r['determined_fraction']} vs {r['valence_determined_fraction']}), which "
            f"contradicts the shared per-volume mechanism the paper gives"
        )


def test_the_limitation_is_stated_as_coupling_not_as_a_gap():
    """The paper must not claim the experiment is merely pending."""
    p = _p()
    for name in ("paper/results.md", "paper/discussion.md",
                 "paper/method_results_walkthrough.md"):
        f = ROOT / name
        if not f.exists():
            continue
        txt = re.sub(r"\s+", " ", f.read_text())
        if "serialization" not in txt:
            continue
        assert "coupled" in txt or "coupling" in txt, (
            f"{name} discusses the serialization limitation without stating the coupling, "
            f"which is what the measurement established"
        )


def test_verdict_matches_the_measured_split():
    p = _p()
    rows = {r["order"]: r for r in p["rows"]}
    if rows["comp_first"]["determined_fraction"] <= 1e-9:
        cons = p["consequence_for_the_requested_experiment"]
        assert cons, "a zero determinability rate must carry a stated consequence"
        # Check the substance rather than a phrase: the consequence must say the arm would
        # have no actionable positions, and must name the coupling as the finding.
        assert "zero actionable positions" in cons
        assert "COUPLED" in cons or "coupled" in cons
        assert "measure the unconstrained model" in cons
    else:
        assert "straightforward" in p["verdict"]
