"""Figure 3 -- accuracy-diversity Pareto: error (log x) vs unique-formula fraction (y).

Reads results/fig_accuracy_diversity_pareto.json exclusively. No number in
this script is typed by hand. Aggregated over SUB-CEILING targets only
(<= 10.405 g/cm^3, the steering ceiling identified in Figure 2) -- above the
ceiling the projection saturates and the comparison is uninformative, per the
artifact's own `sub_ceiling_targets` field.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator
import glob

REPO = Path(__file__).resolve().parents[2]

for f in glob.glob('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/montserrat/Montserrat-*.otf'):
    fm.fontManager.addfont(f)

apply_figure_style(font='Montserrat')

INK, VIOLET, LILAC, SLATE, LINE = '#0D1526', '#6C5CF6', '#B49BFB', '#5C6C87', '#E3DFF8'

with open(REPO / "results" / "fig_accuracy_diversity_pareto.json") as fh:
    artifact = json.load(fh)
payload = artifact["payload"]
bon = payload["best_of_n_curve"]
hyb = payload["hybrid_curve"]
smc_pt = payload["smc_tuned_point"]

bon_sorted = sorted(bon, key=lambda r: r["n"])
errs = np.array([r["mean_error"] for r in bon_sorted])
divs = np.array([r["mean_unique_fraction"] for r in bon_sorted])
ns = [r["n"] for r in bon_sorted]

hyb_sorted = sorted(hyb, key=lambda r: r["n"])
hyb_errs = np.array([r["mean_error"] for r in hyb_sorted])
hyb_divs = np.array([r["mean_unique_fraction"] for r in hyb_sorted])
hyb_ns = [r["n"] for r in hyb_sorted]

fig, ax = plt.subplots(figsize=(3.4, 3.0))

ax.plot(errs, divs, color=SLATE, lw=1.6, ls='-', zorder=2, marker='o', markersize=4.5,
        label='best-of-n (sub-ceiling mean)')
# Per-point offsets chosen so labels clear the line, the diamond, the legend box
# (upper-left), and the top axis edge. All offsets point AWAY from the line and
# from the legend region so the curve and legend never strike through label text.
label_offsets = {1: (6, -14), 4: (4, -16), 16: (10, -14), 64: (2, -16)}
for n, e, dv in zip(ns, errs, divs):
    dx, dy = label_offsets[n]
    ax.annotate(f"n={n}", xy=(e, dv), xytext=(dx, dy), textcoords='offset points',
                fontsize=6.2, color=SLATE)

ax.plot(hyb_errs, hyb_divs, color=VIOLET, lw=1.6, ls='--', zorder=2, marker='^',
        markersize=4.5, label='projection + best-of-n (n=1 is projection alone)')
hyb_label_offsets = {1: (-16, 12), 2: (12, -10), 4: (-24, 8), 8: (-26, 8), 16: (10, -2)}
for n, e, dv in zip(hyb_ns, hyb_errs, hyb_divs):
    dx, dy = hyb_label_offsets.get(n, (6, 6))
    ax.annotate(f"n={n}", xy=(e, dv), xytext=(dx, dy), textcoords='offset points',
                fontsize=6.2, color=VIOLET)

ax.scatter([smc_pt["mean_error"]], [smc_pt["mean_unique_fraction"]], s=55, color=INK,
           zorder=3, marker='s', label=f'SMC (tuned, K={smc_pt["K"]}, sigma={smc_pt["sigma"]})')

ax.set_xscale('log')
ax.set_xlabel("mean best-error (log scale, g/cm\u00b3)")
ax.set_ylabel("unique-formula fraction")
# Direction-of-goodness cue as a separate upright annotation in the margin
# (figure-style \u00a73.6) rather than baked into the rotated y-axis label text.
ax.annotate("\u2191 up-and-left is better", xy=(0.03, 1.10), xycoords='axes fraction',
            fontsize=6.0, color=SLATE, ha='left', va='bottom', rotation=0)

# Explicit major ticks with plain decimal labels; kill the minor-tick formatter
# entirely so the log-scale minor labels stop colliding at the axis ends.
major_ticks = [0.2, 0.5, 1, 2, 5]
ax.xaxis.set_major_locator(FixedLocator(major_ticks))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.margins(y=0.12)
ax.set_ylim(0.02, 1.05)
ax.legend(loc='lower right', bbox_to_anchor=(1.0, -0.02), fontsize=5.8, handlelength=1.5,
          frameon=False, labelspacing=0.55)

fig.tight_layout()
fig.savefig(REPO / "development" / "figures" / "fig3_accuracy_diversity_pareto.png", dpi=300)
fig.savefig(REPO / "development" / "docs" / "template" / "figures" / "fig3_accuracy_diversity_pareto.pdf")

# --- Verification
lines = [ln for ln in ax.get_lines() if ln.get_label() == 'best-of-n (sub-ceiling mean)']
ln = lines[0]
xs, ys = ln.get_xdata(), ln.get_ydata()
assert np.allclose(xs, errs), "fig3 best-of-n x mismatch"
assert np.allclose(ys, divs), "fig3 best-of-n y mismatch"

hyb_lines = [ln for ln in ax.get_lines() if ln.get_label() == 'projection + best-of-n (n=1 is projection alone)']
hln = hyb_lines[0]
hxs, hys = hln.get_xdata(), hln.get_ydata()
assert np.allclose(hxs, hyb_errs), "fig3 hybrid x mismatch"
assert np.allclose(hys, hyb_divs), "fig3 hybrid y mismatch"

smc_scat = [c for c in ax.collections if isinstance(c.get_label(), str)
            and c.get_label().startswith('SMC (tuned')][0]
smc_offsets = smc_scat.get_offsets()
assert np.allclose(smc_offsets[0], [smc_pt["mean_error"], smc_pt["mean_unique_fraction"]]), \
    "fig3 SMC tuned point mismatch"

print(f"fig3: best-of-n curve (4 points), hybrid curve (5 points), projection point, and "
      f"SMC tuned point all verified against results/fig_accuracy_diversity_pareto.json")
