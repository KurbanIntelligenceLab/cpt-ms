"""Tests for the joint-constraint solve and the attainability map."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from scripts.layer.projection import project  # noqa: E402
from scripts.layer.verify_multiconstraint import (  # noqa: E402
    _attainable_range,
    attainability_grid,
    joint_solve_case,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_joint_solve_is_exact_at_several_constraint_counts():
    """Residual must stay at solver precision as constraints are added."""
    for m in (1, 2, 3, 4):
        row = joint_solve_case(B=6, V=128, m=m, seed=0)
        assert row["all_converged"], f"m={m} failed to converge"
        assert row["max_abs_residual"] < 1e-9, (
            f"m={m} residual {row['max_abs_residual']:.2e} is not solver precision"
        )
        assert row["simplex_error"] < 1e-12, f"m={m} output is not a distribution"
        assert row["min_mass"] >= 0.0, f"m={m} produced negative mass"


def test_hessian_conditioning_is_reported_and_finite():
    """A converged solve on a near-singular Hessian is not trustworthy."""
    for m in (2, 3, 4):
        row = joint_solve_case(B=6, V=128, m=m, seed=1)
        assert "hessian_cond_max" in row
        assert np.isfinite(row["hessian_cond_max"]), f"m={m} Hessian condition not finite"


def test_collinear_constraints_are_not_silently_accepted():
    """Two identical constraint rows make the multiplier solve ill-posed."""
    g = torch.Generator().manual_seed(0)
    V = 64
    logits = torch.randn(1, V, generator=g, dtype=torch.float64)
    row = torch.randn(V, generator=g, dtype=torch.float64)
    A = torch.stack([row, row.clone()]).unsqueeze(0)
    p = torch.softmax(logits, -1)
    b0 = torch.einsum("bmv,bv->bm", A, p)
    res = project(logits, A, b0, None, max_iter=100, tol=1e-12)
    q = res.q
    mu = A[0] @ q[0]
    cov = (A[0] * q[0]) @ A[0].T - torch.outer(mu, mu)
    cond = float(torch.linalg.cond(cov))
    assert cond > 1e6 or not bool(res.converged.all()), (
        f"collinear constraints gave condition number {cond:.2e} and converged="
        f"{bool(res.converged.all())}; the ill-posedness is invisible"
    )


def test_attainable_range_bounds_are_vertex_values():
    """The range of a linear functional over a simplex is its coefficient min and max."""
    a = torch.tensor([3.0, -1.0, 7.5, 0.0])
    mask = torch.tensor([True, True, True, False])
    lo, hi = _attainable_range(a, mask)
    assert (lo, hi) == (-1.0, 7.5)


def test_grid_points_lie_inside_each_marginal_range():
    """Every grid point must be individually attainable by construction."""
    g = attainability_grid(V=32, seed=0, n_grid=5)
    lo1, hi1 = g["mass_attainable_range"]
    lo2, hi2 = g["charge_attainable_range"]
    for cell in g["cells"]:
        assert lo1 <= cell["mass_target"] <= hi1
        assert lo2 <= cell["charge_target"] <= hi2
        assert cell["regime"] != "neither", (
            "a grid point was outside its own attainable range, which the construction "
            "is supposed to prevent"
        )


def test_joint_infeasibility_exists_at_small_vocabulary():
    """The attainability map must be able to report infeasibility at all."""
    g = attainability_grid(V=8, seed=0, n_grid=9)
    n = len(g["cells"])
    frac = g["regime_counts"].get("individually_only", 0) / n
    assert frac > 0.0, (
        "no jointly-infeasible cell at V=8; the attainability map cannot distinguish "
        "joint from individual feasibility"
    )


def test_attainability_fraction_is_grid_dependent_and_says_so():
    """The joint fraction changes with grid resolution, so it must carry that caveat."""
    coarse = attainability_grid(V=64, seed=0, n_grid=5)
    fine = attainability_grid(V=64, seed=0, n_grid=15)
    fc = coarse["regime_counts"].get("jointly_attainable", 0) / len(coarse["cells"])
    ff = fine["regime_counts"].get("jointly_attainable", 0) / len(fine["cells"])
    assert ff > fc, (
        f"expected the joint fraction to rise with grid resolution, got {fc:.3f} at "
        f"n_grid=5 and {ff:.3f} at 15"
    )
    art = RESULTS / "multiconstraint.json"
    if art.exists():
        payload = json.loads(art.read_text())["payload"]
        assert payload["summary"]["fractions_are_grid_dependent"] is True
        assert "grid" in payload["vocabulary_sweep"]["caveat"].lower()


def test_composition_gets_easier_with_vocabulary_size():
    """Joint attainability must be reported as a function of vocabulary, not a scalar."""
    art = RESULTS / "multiconstraint.json"
    if not art.exists():
        pytest.skip("multiconstraint artifact absent; run scripts.layer.verify_multiconstraint")
    payload = json.loads(art.read_text())["payload"]
    rows = payload["vocabulary_sweep"]["rows"]
    assert len(rows) >= 4, "sweep too short to establish a trend"
    sizes = [r["vocab_size"] for r in rows]
    assert sizes == sorted(sizes), "sweep rows are not ordered by vocabulary size"
    fracs = [r["jointly_attainable_fraction"] for r in rows]
    assert fracs[0] < fracs[-1], (
        f"joint attainability did not increase with vocabulary: {fracs[0]:.3f} at "
        f"V={sizes[0]} against {fracs[-1]:.3f} at V={sizes[-1]}"
    )
    assert all(b >= a - 1e-9 for a, b in zip(fracs, fracs[1:])), (
        f"joint attainability is not monotone in vocabulary size: {fracs}"
    )
