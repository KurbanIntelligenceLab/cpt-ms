"""Is one retargeted checkpoint EQUIVALENT to seven trained models, or merely not-different?

WHY THIS EXISTS. The amortization claim rests on a paired comparison that returns p = 1.0.
A large p-value is not evidence of equivalence -- it is the absence of evidence of
difference, and with seven paired targets a reviewer is right to ask whether the test could
have detected anything at all. This script answers that question three ways rather than
asserting it: the confidence interval on the paired difference, the minimum effect the
design could detect, and a two-one-sided-tests procedure against a pre-specified margin.

THE MARGIN IS NOT CHOSEN TO PASS. It is the retargeted arm's own mean within-target seed
standard deviation. A difference smaller than the method's own run-to-run reproducibility
is not a difference anyone could act on, and using the arm's own noise as the margin cannot
be tuned after seeing the result without changing the arm. Conventional margins are also
reported so a reader can see the conclusion is not margin-sensitive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/equivalence_test.py"


def main(results_dir: str | Path = "results", smoke: bool = False) -> Path:
    rd = Path(results_dir)
    rep = json.loads((rd / "WS4_retargeting_seed_replication.json").read_text())["payload"]
    rows = rep["paired_vs_frontier"]

    ret = np.array([r["retarget_mean_error"] for r in rows], dtype=float)
    fro = np.array([r["frontier_error"] for r in rows], dtype=float)
    sds = np.array([r["retarget_sd"] for r in rows], dtype=float)
    d = ret - fro
    n = len(d)

    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    ci = (float(d.mean() - tc * se), float(d.mean() + tc * se))
    mde = float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1))
                * d.std(ddof=1) / np.sqrt(n))

    margin = float(sds.mean())          # the arm's own reproducibility
    def tost(m: float) -> float:
        lo = stats.ttest_1samp(d, -m, alternative="greater").pvalue
        hi = stats.ttest_1samp(d, m, alternative="less").pvalue
        return float(max(lo, hi))

    p_primary = tost(margin)
    sensitivity = {f"{m:.2f}": tost(m) for m in (0.10, 0.15, 0.20, 0.25)}

    w = stats.wilcoxon(ret, fro)
    t = stats.ttest_rel(ret, fro)

    payload = {
        "question": ("Is one retargeted checkpoint equivalent to per-target training, or "
                     "is the null result simply underpowered?"),
        "n_paired_targets": int(n),
        "paired_differences": [float(x) for x in d],
        "mean_paired_difference": float(d.mean()),
        "sd_paired_difference": float(d.std(ddof=1)),
        "ci95_on_mean_difference": list(ci),
        "difference_tests": {
            "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "paired_t": float(t.statistic), "paired_t_p": float(t.pvalue),
        },
        "power": {
            "min_detectable_effect_80pct": mde,
            "as_fraction_of_frontier_error": mde / rep["summary"]["frontier_sub_ceiling_mae"],
            "frontier_mean_error": rep["summary"]["frontier_sub_ceiling_mae"],
        },
        "equivalence": {
            "margin_definition": ("mean within-target seed sd of the retargeted arm; a "
                                  "difference below the method's own reproducibility is "
                                  "not actionable"),
            "margin": margin,
            "tost_p": p_primary,
            "equivalent_at_margin": bool(p_primary < 0.05),
            "sensitivity_to_margin": sensitivity,
        },
        "verdict": None,
    }
    if p_primary < 0.05 and ci[0] > -0.25 and ci[1] < 0.25:
        payload["verdict"] = (
            f"Equivalence established at a margin of {margin:.3f} g/cm^3 (TOST p = "
            f"{p_primary:.4f}). The 95% CI on the paired difference is "
            f"[{ci[0]:.3f}, {ci[1]:.3f}], so the claim is a bounded equivalence rather "
            f"than an unrejected null. The design could have detected {mde:.3f} g/cm^3 at "
            f"80% power, which is "
            f"{mde / rep['summary']['frontier_sub_ceiling_mae'] * 100:.0f}% of the "
            f"per-target arm's own mean error."
        )
    else:
        payload["verdict"] = (
            f"Equivalence NOT established at margin {margin:.3f} (TOST p = {p_primary:.4f}). "
            f"The paired comparison is consistent with no difference but the design cannot "
            f"exclude effects up to {mde:.3f} g/cm^3."
        )

    return emit("WS4_equivalence" + ("_smoke" if smoke else ""), payload,
                config={"results_dir": str(rd), "margin_rule": "mean retarget seed sd"},
                seeds=[0, 1, 2], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(results_dir=a.results_dir, smoke=a.smoke))
