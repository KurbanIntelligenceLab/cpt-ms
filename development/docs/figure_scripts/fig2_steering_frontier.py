"""Figure 2 -- requested vs achieved density, seeds as error bars, ceiling marked.

Reads results/fig_steering_frontier.json exclusively. No number in this
script is typed by hand. Aggregation is seed-averaged per target (see the
artifact's own `aggregation` field and reports/steering_frontier_crystal_notes.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
import glob

REPO = Path(__file__).resolve().parents[3]

for f in glob.glob('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/montserrat/Montserrat-*.otf'):
    fm.fontManager.addfont(f)

apply_figure_style(font='Montserrat')

INK, VIOLET, LILAC, SLATE, LINE = '#0D1526', '#6C5CF6', '#B49BFB', '#5C6C87', '#E3DFF8'

with open(REPO / "results" / "fig_steering_frontier.json") as fh:
    artifact = json.load(fh)
payload = artifact["payload"]
frontier = payload["frontier"]
ceiling = payload["steering_ceiling"]

targets = np.array([r["target"] for r in frontier])
means = np.array([r["deployed_mean"] for r in frontier])
sds = np.array([r["deployed_sd"] for r in frontier])

lo, hi = min(targets.min(), means.min()), max(targets.max(), means.max())
pad = 0.06 * (hi - lo)
lims = (lo - pad, hi + pad)

fig, ax = plt.subplots(figsize=(3.4, 3.0))

ax.plot(lims, lims, color=LINE, lw=1.2, ls='--', zorder=1, label='identity (perfect steering)')
ax.errorbar(targets, means, yerr=sds, fmt='o', color=VIOLET, ecolor=VIOLET,
            elinewidth=1.2, capsize=2.5, markersize=4.5, zorder=3,
            label='deployed density (mean \u00b1 seed sd)')

ax.scatter([ceiling["target"]], [ceiling["achieved_density"]], s=70, facecolors='none',
           edgecolors=SLATE, linewidths=1.6, zorder=4)
ax.annotate(f"steering ceiling\n{ceiling['achieved_density']:.2f} g/cm\u00b3",
            xy=(ceiling["target"], ceiling["achieved_density"]),
            xytext=(ceiling["target"] - 0.42 * (lims[1] - lims[0]),
                    ceiling["achieved_density"] - 0.16 * (lims[1] - lims[0])),
            fontsize=6.2, color=SLATE,
            arrowprops=dict(arrowstyle='-', color=SLATE, lw=0.9))

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("requested density (g/cm\u00b3)")
ax.set_ylabel("achieved density (g/cm\u00b3)")
ax.legend(loc='upper left', fontsize=6.0, handlelength=1.6, frameon=False)

fig.tight_layout()
fig.savefig(REPO / "development" / "figures" / "fig2_steering_frontier.png", dpi=300)
fig.savefig(REPO / "development" / "docs" / "template" / "figures" / "fig2_steering_frontier.pdf")

# --- Verification
errbar_line = None
for coll in ax.containers:
    pass

# Extract the errorbar data via the ErrorbarContainer
eb_container = [c for c in ax.containers if hasattr(c, 'lines')][0]
data_line = eb_container.lines[0]
xs, ys = data_line.get_xdata(), data_line.get_ydata()
assert np.allclose(np.sort(xs), np.sort(targets)), "fig2 x mismatch"
order = np.argsort(xs)
assert np.allclose(np.array(ys)[order], means[np.argsort(targets)]), "fig2 y mismatch"

# yerr caplines / bars: check the vertical error bar segments extents
verr_segments = eb_container.lines[2][0]  # first (only) LineCollection: vertical errorbars
segs = verr_segments.get_segments()
seg_by_x = {}
for seg in segs:
    x0 = seg[0][0]
    seg_by_x.setdefault(round(x0, 6), []).append(seg)
n_err_verified = 0
for t, m, s in zip(targets, means, sds):
    key = round(float(t), 6)
    if key in seg_by_x:
        seg = seg_by_x[key][0]
        y_lo, y_hi = seg[0][1], seg[1][1]
        assert np.allclose([y_lo, y_hi], [m - s, m + s], atol=1e-8), f"errbar mismatch at target {t}"
        n_err_verified += 1

print(f"fig2: {len(order)}/{len(targets)} point series verified, {n_err_verified}/{len(targets)} error bars verified against results/fig_steering_frontier.json")
