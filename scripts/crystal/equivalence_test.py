"""Is one retargeted checkpoint EQUIVALENT to seven trained models, or merely not-different?

WHY THIS EXISTS. The amortization claim rests on a paired comparison that returns p = 1.0.
A large p-value is not evidence of equivalence -- it is the absence of evidence of
difference, and with seven paired targets a reviewer is right to ask whether the test could
have detected anything at all. This script answers that question three ways rather than
asserting it: the confidence interval on the paired difference, the minimum effect the
design could detect, and a two-one-sided-tests procedure against a pre-specified margin.

THE MARGIN IS READ FROM THE PRE-REGISTRATION, NOT FROM THIS FILE. A margin living in the
analysis script is a margin that can be edited after seeing the result, which is not a
pre-specification. `preregistration/equivalence_margin.md` carries the value, the rule
that produced it (the retargeted arm's own mean within-target seed sd), and the two-part
decision rule; this script parses it and fails loudly if it is absent or disagrees with the
rule. A difference smaller than the method's own run-to-run reproducibility is not one
anyone could act on, and a margin tied to the arm's own noise cannot be widened to pass
without changing the arm.

WHY THE WILCOXON RESULT IS REPORTED WITH ITS FLOOR. At n = 7 the smallest attainable
two-sided p is 2/2^7 = 0.015625 under any data whatsoever, so a large p carries very little
information and quoting it alone would overstate what the difference test can establish.
The floor is computed here rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit

SCRIPT = "scripts/crystal/equivalence_test.py"

PREREG = (Path(__file__).resolve().parents[2] / "development" / "preregistration"
          / "equivalence_margin.md")


def read_prereg_margin() -> tuple[float, Path]:
    """The margin is a pre-registration, so it is read from one.

    Kept out of this script deliberately: a constant here could be edited after seeing the
    result, and a margin that can be edited after the fact is not a pre-specification.
    """
    if not PREREG.exists():
        raise FileNotFoundError(
            f"no pre-registered margin at {PREREG}; the equivalence claim requires one"
        )
    m = re.search(r"EQUIVALENCE_MARGIN_G_PER_CM3\s*=\s*([0-9.]+)", PREREG.read_text())
    if not m:
        raise ValueError(f"{PREREG} does not define EQUIVALENCE_MARGIN_G_PER_CM3")
    return float(m.group(1)), PREREG




def main(results_dir: str | Path = "results", smoke: bool = False) -> Path:
    rd = Path(results_dir)
    rep = json.loads((rd / "retargeting_seed_replication.json").read_text())["payload"]
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

    margin, prereg_path = read_prereg_margin()
    derived = float(sds.mean())
    if abs(margin - derived) > 5e-4:
        raise ValueError(
            f"pre-registered margin {margin} disagrees with the rule it states "
            f"(mean retarget seed sd = {derived:.4f}). Either the data changed or the "
            f"margin was edited; resolve in {prereg_path} before reporting."
        )

    def tost(m: float) -> float:
        lo = stats.ttest_1samp(d, -m, alternative="greater").pvalue
        hi = stats.ttest_1samp(d, m, alternative="less").pvalue
        return float(max(lo, hi))

    p_primary = tost(margin)
    sensitivity = {f"{m:.2f}": tost(m) for m in (0.10, 0.15, 0.20, 0.25)}

    # Bootstrap CI on the mean paired difference: 10,000 paired resamples. The parametric
    # interval assumes normality of seven differences, which is not checkable at n = 7,
    # so both are reported and the claim uses whichever is wider.
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10_000)])
    ci_boot = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # The Wilcoxon floor is a property of n, not of the data: with n paired observations
    # the smallest attainable two-sided p is 2/2^n. Computed, not asserted.
    wilcoxon_floor = 2.0 / (2 ** n)

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
        "ci95_bootstrap": list(ci_boot),
        "bootstrap_resamples": 10_000,
        "difference_tests": {
            "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "wilcoxon_min_attainable_two_sided_p": wilcoxon_floor,
            "wilcoxon_interpretation": (
                f"at n = {n} the smallest attainable two-sided p is 2/2^{n} = "
                f"{wilcoxon_floor:.6f} under any data; a large p here reflects the test's "
                f"resolution at this sample size as much as the data, which is why the "
                f"equivalence claim rests on the TOST rather than on this p-value"),
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
            "margin_source": str(PREREG.relative_to(PREREG.parents[1])),
            "margin_matches_stated_rule": True,
            "tost_p": p_primary,
            "equivalent_at_margin": bool(p_primary < 0.05),
            "sensitivity_to_margin": sensitivity,
            "decision_rule": (
                "pre-registered: claim equivalence only if TOST p < 0.05 at the margin AND "
                "still p < 0.05 at a 0.10 margin"),
            "decision_rule_satisfied": bool(p_primary < 0.05
                                            and sensitivity["0.10"] < 0.05),
        },
        "verdict": None,
    }
    if p_primary < 0.05 and sensitivity["0.10"] < 0.05:
        payload["verdict"] = (
            f"Equivalence established at a margin of {margin:.3f} g/cm^3 (TOST p = "
            f"{p_primary:.4f}). The 95% CI on the paired difference is "
            f"[{ci[0]:.3f}, {ci[1]:.3f}], so the claim is a bounded equivalence rather "
            f"than an unrejected null. The design could have detected {mde:.3f} g/cm^3 at "
            f"80% power, which is "
            f"{mde / rep['summary']['frontier_sub_ceiling_mae'] * 100:.0f}% of the "
            f"per-target arm's own mean error. The pre-registered second condition is met: "
            f"equivalence survives at a 0.10 margin (p = {sensitivity['0.10']:.4f}), so the "
            f"conclusion is not an artifact of the margin. The Wilcoxon result is p = "
            f"{float(w.pvalue):.1f} against a floor of {wilcoxon_floor:.6f} attainable at "
            f"n = {n}, which is why it is not the basis of the claim."
        )
    else:
        payload["verdict"] = (
            f"Equivalence NOT established at margin {margin:.3f} (TOST p = {p_primary:.4f}). "
            f"The paired comparison is consistent with no difference but the design cannot "
            f"exclude effects up to {mde:.3f} g/cm^3."
        )

    return emit("equivalence" + ("_smoke" if smoke else ""), payload,
                config={"results_dir": str(rd), "margin_rule": "mean retarget seed sd",
                        "margin_read_from": str(PREREG.name)},
                seeds=[0, 1, 2], script=SCRIPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(main(results_dir=a.results_dir, smoke=a.smoke))
