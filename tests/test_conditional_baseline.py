"""The conditional-training baseline must be fair, and coverage must be measured."""

from __future__ import annotations

import numpy as np
import pytest

from _artifacts import payload
from _devtree import requires_mp20
from cpt.crystal.mp20 import load_split
from cpt.train.train_crystal import CrystalTrainConfig, _coverage

pytestmark = requires_mp20


def test_filter_retains_a_trainable_subset_at_every_swept_target():
    """The conditional baseline must be runnable wherever it is compared, or the comparison has a hole."""
    d = np.array([c.density for c in load_split("train")])
    for t in (1.5, 3.0, 5.0, 8.0, 10.0):
        n = int((np.abs(d - t) <= 0.5).sum())
        assert n >= 100, f"target {t} leaves only {n} structures for conditional"


def test_extreme_targets_starve_the_conditional_baseline():
    """Pins WHY conditional cannot simply be run everywhere: the filter empties out."""
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
    mask_only, conditional = CrystalTrainConfig(arm="mask_only"), CrystalTrainConfig(arm="conditional")
    assert conditional.steps == mask_only.steps and conditional.batch == mask_only.batch and conditional.lr == mask_only.lr


def _element_freq(items):
    from collections import Counter
    c = Counter()
    for x in items:
        for e, n in x.formula.items():
            c[e] += n
    tot = sum(c.values())
    return {e: k / tot for e, k in c.items()}


def _tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def test_element_tv_is_a_shift_measure_not_a_diversity_measure():
    """TV against the full data cannot separate the arms, and must not be cited as if it does."""
    train = load_split("train")
    full = _element_freq(train)
    for t in (3.0, 5.0, 8.0, 10.0):
        sl = [c for c in train if abs(c.density - t) <= 0.5]
        assert len(sl) >= 100, f"target {t} slice too small to test"
        assert _tv(_element_freq(sl), full) > 0.15, (
            f"target {t}: the filtered slice's own TV is small, so this test's premise "
            "does not hold there"
        )


def test_conditional_arm_tv_matches_its_own_slice():
    """The conditional arm reproducing its slice is what its TV actually shows."""
    train = load_split("train")
    full = _element_freq(train)
    for t, measured in ((3.0, 0.3919), (5.0, 0.2028)):
        sl = [c for c in train if abs(c.density - t) <= 0.5]
        slice_tv = _tv(_element_freq(sl), full)
        assert abs(slice_tv - measured) < 0.02, (
            f"target {t}: slice TV {slice_tv:.4f} vs the arm's measured {measured:.4f}; "
            "if these diverge the reproduction reading no longer holds"
        )


def test_sparse_regime_claim_is_coverage_not_accuracy():
    """The conditional baseline's sparse-target weakness is coverage, not accuracy."""
    from pathlib import Path
    art = Path("results/conditional_sparse.json")
    if not art.exists():
        pytest.skip("sparse artifact absent; run the sparse-target job")
    p = payload(art.name)
    for t, tgt in (("13.0", 13.0), ("16.0", 16.0)):
        rows = p["rows"][t]
        err = abs(float(np.mean([r["density"] for r in rows])) - tgt)
        assert err < 0.2, (
            f"target {t}: conditional error {err:.3f}; the sparse regime's finding is that "
            "accuracy HOLDS here, so a large error would invalidate the claim"
        )
        assert all(r["valid"] == 1.0 for r in rows), f"target {t}: validity below ceiling"
    cv = p["coverage_vs_slice_size"]
    assert cv["corr_log_n_unique"] > 0.9, (
        f"unique-formula fraction correlates {cv['corr_log_n_unique']:.3f} with log slice "
        "size; the coverage-collapse claim rests on this being strong and positive"
    )
    assert p["refusal"]["n_structures"] < p["config"]["floor"] if "config" in p else True


def test_only_one_swept_target_actually_starves():
    """The refusal is a single case, and overstating it was the original error."""
    from pathlib import Path
    art = Path("results/conditional_sparse.json")
    if not art.exists():
        pytest.skip("sparse artifact absent")
    ref = payload(art.name)["refusal"]
    assert ref["target"] == 20.0 and ref["n_structures"] == 6
    d = np.array([c.density for c in load_split("train")])
    starved = [t for t in (1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0, 13.0, 16.0, 20.0)
               if int((np.abs(d - t) <= 0.5).sum()) < ref["floor"]]
    assert starved == [20.0], f"expected exactly one starved target, got {starved}"
