"""Analyze the A7 two-constraint grid: does a second additive constraint steer?

WHAT THIS TESTS. A7 trains through a JOINT projection of mass density and valence
electron density. The grid crosses three density targets with three valence targets, and
each row is accompanied by a single-constraint A3 control at the same density target, so
any difference is attributable to the second constraint rather than to training noise.

THE DECISIVE QUANTITY IS PASS-THROUGH, NOT ERROR. A dial that reports a plausible value
at every target has not necessarily been steered: it may simply be reproducing whatever
the unconstrained model does. So the measure here is the RANGE of achieved values divided
by the range of requested values, checked against the control, and monotonicity in the
target. A non-monotone response means the dial is not being controlled at all, however
small the apparent errors.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/analyze_a7_grid.py"
DATA_VALENCE_SD = 0.2019   # CP7_second_channel.json channel_statistics
DATA_VALENCE_MEAN = 0.4839


def load(results_dir: Path) -> tuple[list[dict], dict]:
    cells = []
    for p in sorted(glob.glob(str(results_dir / "A7grid_d*.json"))):
        j = json.load(open(p))["payload"]
        f = j["final"]
        d = f.get("deployed") or {}
        jp = j.get("joint_projection") or {}
        cells.append({
            "density_target": j["train_config"]["density_target"],
            "valence_target": j.get("valence_target"),
            "deployed_density": d.get("mean_density"),
            "deployed_valence": d.get("mean_valence_density"),
            "backbone_density": f.get("mean_density"),
            "backbone_valence": f.get("mean_valence_density"),
            "validity": f.get("structural_validity_rate"),
            "unique_formula_fraction": f.get("unique_formula_fraction"),
            "novel_formula_fraction": f.get("novel_formula_fraction"),
            "n_actionable": jp.get("n_actionable_positions"),
            "n_joint": jp.get("n_joint_positions"),
            "n_jointly_infeasible": jp.get("n_jointly_infeasible"),
            "n_budget_exhausted": jp.get("n_excluded_budget_exhausted"),
            "fallback_rate": jp.get("joint_fallback_rate"),
            "source": Path(p).name,
        })
    controls = {}
    for p in sorted(glob.glob(str(results_dir / "A7grid_control_A3_d*.json"))):
        j = json.load(open(p))["payload"]
        f = j["final"]
        d = f.get("deployed") or {}
        controls[j["train_config"]["density_target"]] = {
            "deployed_density": d.get("mean_density"),
            "deployed_valence": d.get("mean_valence_density"),
            "backbone_density": f.get("mean_density"),
            "validity": f.get("structural_validity_rate"),
            "unique_formula_fraction": f.get("unique_formula_fraction"),
            "source": Path(p).name,
        }
    return cells, controls


def per_row(cells: list[dict], controls: dict) -> list[dict]:
    """Pass-through and monotonicity for each density target with >=2 valence cells."""
    out = []
    by_d: dict[float, list[dict]] = {}
    for c in cells:
        by_d.setdefault(c["density_target"], []).append(c)
    for dt, rows in sorted(by_d.items()):
        rows = sorted(rows, key=lambda r: r["valence_target"])
        if len(rows) < 2:
            out.append({"density_target": dt, "n_cells": len(rows),
                        "complete": False,
                        "note": "fewer than two valence cells; pass-through undefined"})
            continue
        vt = np.array([r["valence_target"] for r in rows], dtype=float)
        va = np.array([r["deployed_valence"] for r in rows], dtype=float)
        fb = np.array([r["fallback_rate"] for r in rows], dtype=float)
        nj = np.array([r["n_joint"] for r in rows], dtype=float)
        ctrl = controls.get(dt) or {}
        cv = ctrl.get("deployed_valence")
        req = float(vt.max() - vt.min())
        ach = float(va.max() - va.min())
        out.append({
            "density_target": dt,
            "n_cells": len(rows),
            "complete": len(rows) == 3,
            "valence_targets": [float(x) for x in vt],
            "valence_achieved": [float(x) for x in va],
            "requested_range": req,
            "achieved_range": ach,
            "pass_through": (ach / req if req else None),
            "monotone_in_target": bool(np.all(np.diff(va) > 0)),
            "control_valence": cv,
            "max_deviation_from_control": (float(np.max(np.abs(va - cv)))
                                           if cv is not None else None),
            "max_deviation_in_data_sd": (float(np.max(np.abs(va - cv)) / DATA_VALENCE_SD)
                                         if cv is not None else None),
            "fallback_rates": [float(x) for x in fb],
            "corr_target_fallback": (float(np.corrcoef(vt, fb)[0, 1])
                                     if len(vt) > 2 else None),
            "corr_target_joint_positions": (float(np.corrcoef(vt, nj)[0, 1])
                                            if len(vt) > 2 else None),
            "joint_positions": [int(x) for x in nj],
            "density_achieved": [float(r["deployed_density"]) for r in rows],
            "density_control": ctrl.get("deployed_density"),
            "validity_values": sorted({r["validity"] for r in rows}),
        })
    return out


def main(results_dir: str | Path = "results", smoke: bool = False) -> Path:
    rd = Path(results_dir)
    cells, controls = load(rd)
    if not cells:
        raise SystemExit("no A7 grid cells found in %s" % rd)
    rows = per_row(cells, controls)
    complete = [r for r in rows if r.get("complete")]

    # MATCHED TARGETS ONLY. Averaging A7 over the density targets it reached against
    # controls over all three is confounded: the d=8.0 control sits above the reachable
    # ceiling and carries an error of 1.26, so an unmatched comparison would credit A7
    # with an advantage that is really the absence of that cell. Only density targets
    # present in BOTH arms are compared.
    have_a7 = {c["density_target"] for c in cells if c["deployed_density"] is not None}
    matched = sorted(t for t in have_a7 if controls.get(t, {}).get("deployed_density"))
    dens_err = [abs(c["deployed_density"] - c["density_target"]) for c in cells
                if c["deployed_density"] is not None
                and c["density_target"] in matched]
    ctrl_err = [abs(controls[t]["deployed_density"] - t) for t in matched]

    # FIXED-TARGET COLUMN. A row needs two valence cells for pass-through, so a partial row
    # contributes nothing to that measure. But the cells at a SHARED valence target across
    # density rows are directly comparable, and they test the constraint's effect more
    # cleanly than pass-through does: each cell is paired with its own single-constraint
    # control, so the comparison is deviation-from-control rather than an absolute level
    # that the density constraint drags around on its own.
    by_vt: dict[float, list[dict]] = {}
    for c in cells:
        by_vt.setdefault(c["valence_target"], []).append(c)
    columns = []
    for vt, cs in sorted(by_vt.items()):
        if len(cs) < 2:
            continue
        rows_c = []
        for c in sorted(cs, key=lambda x: x["density_target"]):
            ctrl = controls.get(c["density_target"]) or controls.get(str(c["density_target"])) or {}
            ctrl_v = ctrl.get("deployed_valence")
            if ctrl_v is None or c["deployed_valence"] is None:
                continue
            asked_down = vt < ctrl_v
            moved_down = c["deployed_valence"] < ctrl_v
            rows_c.append({
                "density_target": c["density_target"],
                "valence_achieved": c["deployed_valence"],
                "control_valence": ctrl_v,
                "deviation_from_control": c["deployed_valence"] - ctrl_v,
                "asked_direction": "down" if asked_down else "up",
                "moved_correct_direction": asked_down == moved_down,
            })
        if len(rows_c) < 2:
            continue
        dts = np.array([r["density_target"] for r in rows_c], dtype=float)
        dev = np.array([r["deviation_from_control"] for r in rows_c], dtype=float)
        columns.append({
            "valence_target": vt,
            "n_density_rows": len(rows_c),
            "rows": rows_c,
            "all_moved_correct_direction": all(r["moved_correct_direction"] for r in rows_c),
            "corr_density_target_deviation": (float(np.corrcoef(dts, dev)[0, 1])
                                              if len(rows_c) > 2 else None),
            "note": ("cells at one valence target across density rows, each against its own "
                     "single-constraint control; a partial row still contributes here"),
        })

    summary = {
        "n_cells": len(cells),
        "n_controls": len(controls),
        "n_complete_rows": len(complete),
        "matched_density_targets": matched,
        "density_mae_A7_matched": float(np.mean(dens_err)) if dens_err else None,
        "density_mae_A3_matched": float(np.mean(ctrl_err)) if ctrl_err else None,
        "unmatched_control_targets": sorted(set(controls) - set(matched)),
        "validity_all_cells": sorted({c["validity"] for c in cells}
                                     | {v["validity"] for v in controls.values()}),
        "pass_through_by_row": {str(r["density_target"]): r.get("pass_through")
                                for r in complete},
        "monotone_by_row": {str(r["density_target"]): r.get("monotone_in_target")
                            for r in complete},
        "fixed_valence_columns": columns,
        "verdict": None,
    }
    if complete:
        pt = [r["pass_through"] for r in complete if r["pass_through"] is not None]
        mono = [r["monotone_in_target"] for r in complete]
        summary["max_pass_through"] = max(pt) if pt else None
        summary["any_row_monotone"] = any(mono)
        # The verdict is derived from the measurements, not asserted alongside them.
        if pt and max(pt) < 0.25 and not any(mono):
            summary["verdict"] = (
                "The second constraint does not steer: pass-through below 25% on every "
                "complete row and no row monotone in its target. Density steering is "
                "unaffected, so this is a limit on constraint COMPOSITION for two "
                "correlated additive quantities sharing one token budget, not a failure "
                "of the layer."
            )
        elif pt and max(pt) > 0.5 and all(mono):
            summary["verdict"] = ("The second constraint steers: pass-through above 50% "
                                  "and monotone on every complete row.")
        else:
            summary["verdict"] = ("Mixed: see pass_through_by_row and monotone_by_row; "
                                  "neither the steering nor the no-steering criterion is "
                                  "met on all complete rows.")

    return emit("WS5_two_constraint_grid" + ("_smoke" if smoke else ""),
                {"question": "Does a second additive expectation constraint steer when "
                             "trained jointly with density, and what happens to joint "
                             "feasibility as the second target moves away from the data?",
                 "cells": cells, "controls": controls, "rows": rows,
                 "summary": summary,
                 "data_reference": {"valence_mean": DATA_VALENCE_MEAN,
                                    "valence_sd": DATA_VALENCE_SD}},
                config={"results_dir": str(rd), "smoke": smoke},
                seeds=[0], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(results_dir=a.results_dir, smoke=a.smoke))
