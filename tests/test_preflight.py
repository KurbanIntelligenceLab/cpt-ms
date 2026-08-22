"""Verification for the pre-flight harness."""

from __future__ import annotations

import pytest

from cpt.common.preflight import run_all


def test_preflight_runs_every_check_despite_failure(monkeypatch):
    """One broken check must not hide the state of the others."""
    import cpt.common.preflight as pf

    monkeypatch.setitem(pf.CHECKS, "determinism", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = run_all()
    assert set(rep["checks"]) == set(pf.CHECKS)
    assert rep["checks"]["determinism"]["ok"] is False


def test_preflight_solver_check_catches_a_residual_floor():
    """The regression the team asked for, at harness level."""
    pytest.importorskip("torch")
    from cpt.common.preflight import check_solver

    res = check_solver()
    if res.get("ok") is None:
        pytest.skip(res["skipped"])
    assert res["residual_near_optimum"] < 1e-12
    assert res["ok"] is True
