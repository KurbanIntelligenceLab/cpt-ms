"""Pre-flight smoke-test harness."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit, load
from scripts.common.ledger import finish_run, read_ledger, start_run, unfinished

SCRIPT = "scripts/common/preflight.py"


def check_imports() -> dict:
    """Every module the pipeline needs must import here."""
    mods = [
        "scripts.common.artifacts",
        "scripts.common.ledger",
        "scripts.synthetic.task",
        "scripts.synthetic.targets",
        "scripts.crystal.parser",
        "scripts.crystal.family_ii",
    ]
    failed = {}
    for m in mods:
        try:
            __import__(m)
        except Exception as exc:
            failed[m] = f"{type(exc).__name__}: {exc}"
    optional = {}
    for m in ("scripts.layer.projection",):
        try:
            __import__(m)
        except Exception as exc:
            optional[m] = f"{type(exc).__name__}: {exc}"
    return {"required_failed": failed, "optional_failed": optional, "ok": not failed}


def check_determinism() -> dict:
    """A fixed seed must reproduce bit-identical enumerated quantities."""
    from scripts.synthetic.task import TaskConfig, enumerate_task

    cfg = TaskConfig(length=4, max_count=2)
    a = enumerate_task(cfg)
    b = enumerate_task(cfg)
    same_p = bool((a.log_p_data == b.log_p_data).all())
    same_q = bool((a.total_charge == b.total_charge).all())
    return {"log_p_identical": same_p, "charges_identical": same_q, "ok": same_p and same_q}


def check_ledger() -> dict:
    """Ledger must round-trip in this filesystem."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ledger.jsonl"
        rec = start_run("analysis", SCRIPT, {"probe": True}, [0], ledger_path=path)
        finish_run(rec, "completed", artifacts=["none"], ledger_path=path)
        rows = read_ledger(path)
    ok = len(rows) == 2 and rows[-1]["status"] == "completed"
    return {"records_written": len(rows), "ok": ok}


def check_oracle_coverage() -> dict:
    """Fail if a verification artifact recorded zero oracle comparisons."""
    from scripts.common.artifacts import RESULTS_DIR

    path = RESULTS_DIR / "gradient_verification.json"
    if not path.exists():
        return {"skipped": "gradient_verification.json not yet emitted"}
    tol = json.loads(path.read_text())["payload"]["tolerances"]
    n = tol.get("cvxpylayers_configs_compared", 0)
    return {
        "ok": n > 0,
        "cvxpylayers_configs_compared": n,
        "worst_rel_diff_vs_cvxpylayers": tol.get("worst_rel_diff_vs_cvxpylayers"),
        "detail": (
            "oracle ran" if n > 0
            else "artifact carries no cvxpylayers comparison: re-run after "
                 "installing cvxpylayers, or any cited agreement figure is stale"
        ),
    }


def check_artifacts() -> dict:
    """Artifact emission must round-trip with an intact payload checksum."""
    with tempfile.TemporaryDirectory() as td:
        emit("preflight_probe", {"x": [1, 2, 3]}, config={"probe": True},
             seeds=[0], script=SCRIPT, results_dir=Path(td))
        rec = load("preflight_probe", results_dir=Path(td))
    ok = rec["payload"]["x"] == [1, 2, 3]
    return {"checksum_verified": ok, "ok": ok}


def check_solver() -> dict:
    """Projection must converge to tolerance on this machine."""
    try:
        import torch
        from scripts.layer.projection import project
    except ImportError as exc:
        return {"ok": None, "skipped": f"torch unavailable: {exc}"}

    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 16, generator=g, dtype=torch.float64)
    A = torch.randn(4, 1, 16, generator=g, dtype=torch.float64)
    p = torch.softmax(logits, dim=-1)
    b0 = torch.einsum("bmv,bv->bm", A, p)

    far = project(logits, A, b0 + 0.3)
    near = project(logits, A, b0 + 1e-9)
    exact = project(logits, A, b0)

    worst = max(float(r.residual.abs().max()) for r in (far, near, exact))
    return {
        "residual_far_from_optimum": float(far.residual.abs().max()),
        "residual_near_optimum": float(near.residual.abs().max()),
        "residual_at_optimum": float(exact.residual.abs().max()),
        "iterations": [far.iterations, near.iterations, exact.iterations],
        "worst_residual": worst,
        "ok": worst < 1e-10,
    }


def check_unfinished() -> dict:
    """Runs left in 'started' from a previous crash may have partial artifacts."""
    stale = unfinished()
    return {"count": len(stale), "run_ids": [r["run_id"] for r in stale][:20], "ok": True}


CHECKS: dict[str, Callable[[], dict]] = {
    "imports": check_imports,
    "determinism": check_determinism,
    "ledger": check_ledger,
    "artifacts": check_artifacts,
    "oracle_coverage": check_oracle_coverage,
    "solver": check_solver,
    "unfinished": check_unfinished,
}


def run_all() -> dict:
    """Run every check; never stop at the first failure."""
    results = {}
    for name, fn in CHECKS.items():
        try:
            results[name] = fn()
        except Exception:
            results[name] = {"ok": False, "traceback": traceback.format_exc(limit=4)}
    hard = [n for n, r in results.items() if r.get("ok") is False]
    skipped = [n for n, r in results.items() if r.get("ok") is None]
    return {
        "checks": results,
        "failed": hard,
        "skipped": skipped,
        "all_passed": not hard,
    }


def main(emit_artifact: bool = True) -> dict:
    report = run_all()
    if emit_artifact:
        emit("preflight_report", report,
             config={"checks": sorted(CHECKS)}, seeds=[], script=SCRIPT)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-artifact", action="store_true")
    args = ap.parse_args()
    rep = main(emit_artifact=not args.no_artifact)
    print(json.dumps({k: v.get("ok") for k, v in rep["checks"].items()}, indent=1))
    if rep["skipped"]:
        print("skipped:", rep["skipped"])
    if not rep["all_passed"]:
        print("FAILED:", rep["failed"])
        sys.exit(1)
    print("preflight OK")
