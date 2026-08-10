"""The conditional-training baseline must be fair, and coverage must be measured.

A4 trains on the subset of MP-20 near the target and applies no decode-time
constraint. It is the "just retrain on filtered data" objection, so the comparison
has to be constructed so it cannot be won by an accounting choice:

  - A4 is scored against the FULL data's chemistry, never its own filtered subset,
    since scoring it against the slice it trained on would flatter it.
  - A4 gets the same step budget as A2/A3, which is generous to the baseline: its
    training set is smaller, so it sees each structure more often.
  - Accuracy alone cannot separate the arms, so coverage is measured too.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.crystal.mp20 import ELEMENTS, CrystalVocab, load_split
from scripts.train.train_crystal import CrystalTrainConfig, _coverage


def test_filter_retains_a_trainable_subset_at_every_swept_target():
    """A4 must be runnable wherever it is compared, or the comparison has a hole."""
    d = np.array([c.density for c in load_split("train")])
    for t in (1.5, 3.0, 5.0, 8.0, 10.0):
        n = int((np.abs(d - t) <= 0.5).sum())
        assert n >= 100, f"target {t} leaves only {n} structures for A4"


def test_extreme_targets_starve_the_conditional_baseline():
    """Pins WHY A4 cannot simply be run everywhere: the filter empties out.

    At the extreme targets the filtered training set is too small to train on, which
    is itself the argument — a conditional baseline needs data at the target and a
    steered model does not.
    """
    d = np.array([c.density for c in load_split("train")])
    for t in (16.0, 20.0):
        n = int((np.abs(d - t) <= 0.5).sum())
        assert n < 200, f"target {t} unexpectedly has {n} structures"


def test_coverage_detects_collapse():
    """A sample repeating one formula must not look diverse."""
    from collections import Counter
    collapsed = _coverage(Counter({"Fe": 100, "O": 150}),
                          [(("Fe", 2), ("O", 3))] * 50, set(), {"Fe": 0.1, "O": 0.2})
    assert collapsed["unique_formula_fraction"] == pytest.approx(1 / 50)
    assert collapsed["distinct_elements"] == 2


def test_coverage_reference_is_the_full_data_not_the_subset():
    """TV must be computed against the reference passed in, not the sample itself."""
    from collections import Counter
    same = _coverage(Counter({"Fe": 1, "O": 1}), [(("Fe", 1), ("O", 1))],
                     set(), {"Fe": 0.5, "O": 0.5})
    assert same["element_tv_vs_full_data"] == pytest.approx(0.0, abs=1e-12)
    diff = _coverage(Counter({"Fe": 2}), [(("Fe", 1),)], set(), {"O": 1.0})
    assert diff["element_tv_vs_full_data"] == pytest.approx(1.0)


def test_novelty_is_relative_to_what_the_arm_trained_on():
    """Novel means absent from THIS arm's training set."""
    from collections import Counter
    f = (("Fe", 2), ("O", 3))
    seen = _coverage(Counter({"Fe": 2, "O": 3}), [f], {f}, {"Fe": 0.5, "O": 0.5})
    assert seen["novel_formula_fraction"] == pytest.approx(0.0)
    unseen = _coverage(Counter({"Fe": 2, "O": 3}), [f], set(), {"Fe": 0.5, "O": 0.5})
    assert unseen["novel_formula_fraction"] == pytest.approx(1.0)


def test_a4_step_budget_is_not_reduced():
    """The baseline must not be handicapped by a smaller budget."""
    a2, a4 = CrystalTrainConfig(arm="A2"), CrystalTrainConfig(arm="A4")
    assert a4.steps == a2.steps and a4.batch == a2.batch and a4.lr == a2.lr
