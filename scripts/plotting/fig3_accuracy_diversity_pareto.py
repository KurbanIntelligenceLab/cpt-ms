"""Figure 3 -- accuracy-diversity Pareto: error (log x) vs unique-formula fraction (y).

Reads results/FIG3_accuracy_diversity_pareto.json exclusively. No number in
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

REPO = Path("/Users/jp/cpt-ms")

for f in glob.glob('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/montserrat/Montserrat-*.otf'):
    fm.fontManager.addfont(f)

apply_figure_style(font='Montserrat')

INK, VIOLET, LILAC, SLATE, LINE = '#0D1526', '#6C5CF6', '#B49BFB', '#5C6C87', '#E3DFF8'

with open(REPO / "results" / "FIG3_accuracy_diversity_pareto.json") as fh:
    artifact = json.load(fh)
payload = artifact["payload"]
bon = payload["best_of_n_curve"]
proj = payload["projection_point"]

bon_sorted = sorted(bon, key=lambda r: r["n"])
errs = np.array([r["mean_error"] for r in bon_sorted])
divs = np.array([r["mean_unique_fraction"] for r in bon_sorted])
ns = [r["n"] for r in bon_sorted]

fig, ax = plt.subplots(figsize=(3.4, 3.0))

ax.plot(errs, divs, color=SLATE, lw=1.6, ls='-', zorder=2, marker='o', markersize=4.5,
        label='best-of-n (sub-ceiling mean)')
# Per-point offsets chosen so labels clear the line, the diamond, and the top
# axis edge (n=1 sits highest and rightmost, near the panel corner). All offsets
# point AWAY from the line (below-right or above-right of each marker) so the
# curve never strikes through the label text.
label_offsets = {1: (-30, 8), 4: (8, -14), 16: (-4, -15), 64: (2, 14)}
for n, e, dv in zip(ns, errs, divs):
    dx, dy = label_offsets[n]
    ax.annotate(f"n={n}", xy=(e, dv), xytext=(dx, dy), textcoords='offset points',
                fontsize=6.2, color=SLATE)

ax.scatter([proj["mean_error"]], [proj["mean_unique_fraction"]], s=60, color=VIOLET,
           zorder=3, marker='D', label='projection (deployed A3)')

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
ax.set_ylim(top=ax.get_ylim()[1] + 0.16)
ax.legend(loc='upper left', fontsize=6.2, handlelength=1.6, frameon=False)

fig.tight_layout()
fig.savefig(REPO / "figures" / "fig3_accuracy_diversity_pareto.png", dpi=300)
fig.savefig(REPO / "template" / "figures" / "fig3_accuracy_diversity_pareto.pdf")

# --- Verification
lines = [ln for ln in ax.get_lines() if ln.get_label() == 'best-of-n (sub-ceiling mean)']
ln = lines[0]
xs, ys = ln.get_xdata(), ln.get_ydata()
assert np.allclose(xs, errs), "fig3 best-of-n x mismatch"
assert np.allclose(ys, divs), "fig3 best-of-n y mismatch"

scat = [c for c in ax.collections if c.get_label() == 'projection (deployed A3)'][0]
offsets = scat.get_offsets()
assert np.allclose(offsets[0], [proj["mean_error"], proj["mean_unique_fraction"]]), "fig3 projection point mismatch"

print(f"fig3: best-of-n curve (4 points) verified, projection point verified against results/FIG3_accuracy_diversity_pareto.json")
