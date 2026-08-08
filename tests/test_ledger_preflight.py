"""Verification for the run ledger and the pre-flight harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common.ledger import (
    finish_run,
    latest_status,
    read_ledger,
    start_run,
    unfinished,
)
from scripts.common.preflight import run_all


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "ledger.jsonl"


def test_run_round_trips(ledger):
    rec = start_run("train", "s.py", {"lr": 1e-3}, [0, 1], ledger_path=ledger)
    finish_run(rec, "completed", artifacts=["a.json"], ledger_path=ledger)
    rows = read_ledger(ledger)
    assert [r["status"] for r in rows] == ["started", "completed"]
    assert rows[-1]["artifacts"] == ["a.json"]
    assert rows[0]["run_id"] == rows[1]["run_id"]


def test_started_record_is_not_rewritten(ledger):
    """The started/finished pair is the audit trail; both must survive."""
    rec = start_run("eval", "s.py", {}, [0], ledger_path=ledger)
    finish_run(rec, "failed", ledger_path=ledger)
    rows = read_ledger(ledger)
    assert len(rows) == 2
    assert rows[0]["status"] == "started" and rows[0]["finished_utc"] is None


def test_unfinished_runs_are_detected(ledger):
    a = start_run("train", "s.py", {}, [0], ledger_path=ledger)
    start_run("train", "s.py", {}, [1], ledger_path=ledger)   # never finished
    finish_run(a, "completed", ledger_path=ledger)
    stale = unfinished(ledger_path=ledger)
    assert len(stale) == 1
    assert stale[0]["run_id"] != a.run_id


def test_truncated_final_line_is_tolerated(ledger):
    """A hard kill mid-write must not make earlier records unreadable."""
    rec = start_run("train", "s.py", {}, [0], ledger_path=ledger)
    finish_run(rec, "completed", ledger_path=ledger)
    with open(ledger, "a") as fh:
        fh.write('{"run_id": "truncated", "sta')
    rows = read_ledger(ledger)
    assert len(rows) == 2
    assert rows[-1]["status"] == "completed"


def test_config_hash_is_stable_and_discriminating(ledger):
    a = start_run("train", "s.py", {"lr": 1e-3, "n": 4}, [0], ledger_path=ledger)
    b = start_run("train", "s.py", {"n": 4, "lr": 1e-3}, [0], ledger_path=ledger)
    c = start_run("train", "s.py", {"lr": 2e-3, "n": 4}, [0], ledger_path=ledger)
    assert a.config_hash == b.config_hash      # key order must not matter
    assert a.config_hash != c.config_hash


def test_terminal_status_is_validated(ledger):
    rec = start_run("train", "s.py", {}, [0], ledger_path=ledger)
    with pytest.raises(ValueError):
        finish_run(rec, "started", ledger_path=ledger)


def test_supersede_chain_is_visible(ledger):
    """A correction is a new record, not an edit, so history stays readable."""
    first = start_run("eval", "s.py", {"v": 1}, [0], ledger_path=ledger)
    finish_run(first, "superseded", ledger_path=ledger)
    second = start_run("eval", "s.py", {"v": 2}, [0], supersedes=first.run_id,
                       ledger_path=ledger)
    finish_run(second, "completed", ledger_path=ledger)
    state = latest_status(ledger_path=ledger)
    assert state[first.run_id]["status"] == "superseded"
    assert state[second.run_id]["supersedes"] == first.run_id


def test_preflight_runs_every_check_despite_failure(monkeypatch):
    """One broken check must not hide the state of the others."""
    import scripts.common.preflight as pf

    monkeypatch.setitem(pf.CHECKS, "determinism", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = run_all()
    assert set(rep["checks"]) == set(pf.CHECKS)
    assert rep["checks"]["determinism"]["ok"] is False
    assert rep["checks"]["ledger"]["ok"] is True


def test_preflight_solver_check_catches_a_residual_floor():
    """The regression the team asked for, at harness level.

    The near-optimal-at-initialization case is where the dual is flattest and a
    line-search regression reintroduces a residual floor. The harness must fail
    rather than pass when that happens, because a floor biases the
    active-multiplier fraction that CP0 gates on.
    """
    torch = pytest.importorskip("torch")
    from scripts.common.preflight import check_solver

    res = check_solver()
    if res.get("ok") is None:
        pytest.skip(res["skipped"])
    assert res["residual_near_optimum"] < 1e-12
    assert res["ok"] is True
