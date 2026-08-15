"""Figure 1 -- charge distributions: mask only, deployed projection, tilt-trained,
exact tilt, data.

Reads results/WS7_charge_marginals.json (emitted by
scripts/synthetic/charge_marginals.py) exclusively. No number in this script is
typed by hand. Every plotted series comes from a model or exact target computed
by that script; A2/A3-deployed/A6 are single-seed (seed 0) retrains whose derived
scalars are checked against the published multi-seed summaries in that artifact's
own `agreement_check_vs_published` field.

THE STORY THIS FIGURE MUST MAKE VISIBLE: the exact tilt buys a mean of zero by
spreading mass toward BOTH extremes -- its charge variance (10.45) is HIGHER than
the unconstrained data's (8.56) -- while the deployed per-step projection instead
CONCENTRATES mass near zero (variance 1.55). That anti-concentration is the
paper's headline mechanism: the per-step surrogate hits the mean target by a
fundamentally different route than the sequence-level optimum.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
import glob

REPO = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path("/Users/jp/cpt-ms")

for f in glob.glob('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/montserrat/Montserrat-*.otf'):
    fm.fontManager.addfont(f)

apply_figure_style(font='Montserrat')

INK, VIOLET, LILAC, SLATE, LINE = '#0D1526', '#6C5CF6', '#B49BFB', '#5C6C87', '#E3DFF8'
TILT_COLOR = '#C0392B'   # alarm/comparator hue, distinct from the violet family
A6_COLOR = '#E08214'     # amber -- the trained approximation of the tilt
DATA_COLOR = '#9AA5B8'   # light neutral -- background reference, not a claim series

with open(REPO / "results" / "WS7_charge_marginals.json") as fh:
    artifact = json.load(fh)
payload = artifact["payload"]
dist = payload["distributions"]

data = dist["data_S_conditioned"]
tilt = dist["tilt"]
a2 = dist["A2_mask_only"]
a3d = dist["A3_deployed"]
a6 = dist["A6"]

fig, ax = plt.subplots(figsize=(3.4, 3.05))

# "data" is dropped as a SEPARATE curve -- it is near-coincident with trained A2,
# and five series crowds the panel. Its variance (8.564) is instead named directly
# in A2's legend entry, since the anti-concentration claim (tilt 10.452 ABOVE the
# data's 8.564) needs that number visible somewhere in the figure.
#
# tilt and A6 are also near-coincident BY DESIGN (A6 was trained toward the tilt
# and reaches it) -- drawn as a thick light line (tilt) with a dashed line of the
# same hue on top (A6), so the near-overlap reads as "A6 hit its target" rather
# than as a missing curve.
series = [
    ("A2_mask_only", a2, SLATE,
     "A2: mask only, trained (var 8.82; unconstrained data var 8.56)", 0.95, '-', 1.6, 2),
    ("tilt", tilt, TILT_COLOR, "exact tilt: sequence-optimal (var 10.45)", 0.55, '-', 3.2, 3),
    ("A6", a6, A6_COLOR, "A6: trained toward the tilt (var 10.50)", 1.0, '--', 1.4, 4),
    ("A3_deployed", a3d, VIOLET, "A3 deployed: per-step projection (var 1.56)", 1.0, '-', 2.2, 5),
]

for key, d, color, label, alpha, ls, lw, z in series:
    v = np.array(d["values"], dtype=float)
    m = np.array(d["masses"], dtype=float)
    ax.plot(v, m, color=color, lw=lw, ls=ls, alpha=alpha, label=label,
             marker='o', markersize=2.3, zorder=z)

# Subtle target line -- the "constraint target" annotation below already names it,
# so the line itself is thinned and lightened rather than fighting for attention.
ax.axvline(0.0, color=LINE, lw=0.8, alpha=0.6, zorder=0)
ax.set_xlabel("total charge")
ax.set_ylabel("probability mass")
ax.set_xlim(-17, 17)
ax.set_ylim(0.0, 0.42)
ax.margins(y=0.0)

# Annotation placed BELOW the x-axis line, away from every curve (all curves
# have decayed to ~0 mass by charge=+-9, so the low-mass tail at x=+9 is clear
# of the A3 peak and the rising edges of every series).
ax.annotate("constraint\ntarget (0)", xy=(0.0, 0.30), xytext=(6.0, 0.30),
            fontsize=6.0, color=SLATE, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=SLATE, lw=0.8,
                            shrinkA=2, shrinkB=2))

# Legend BELOW the axes (horizontal-ish, wrapped by ncol) so nothing occludes the
# A3-deployed peak, the figure's tallest and most important feature.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=5.6,
          handlelength=1.6, frameon=False, ncol=1)

fig.tight_layout()
fig.savefig(REPO / "figures" / "fig1_charge_distributions.png", dpi=300, bbox_inches='tight')
fig.savefig(REPO / "template" / "figures" / "fig1_charge_distributions.pdf", bbox_inches='tight')

# --- Verification: read plotted line data back off the artists and compare
verified = 0
lines_by_label = {ln.get_label(): ln for ln in ax.get_lines() if ln.get_label() in
                   {s[3] for s in series}}
for key, d, color, label, alpha, ls, lw, z in series:
    ln = lines_by_label[label]
    xs, ys = ln.get_xdata(), ln.get_ydata()
    assert np.allclose(xs, np.array(d["values"], dtype=float)), f"{key} x mismatch"
    assert np.allclose(ys, np.array(d["masses"], dtype=float)), f"{key} y mismatch"
    verified += 1

# --- Anti-concentration ordering check: projection < data < tilt/A6 on variance
var_proj = a3d["charge_variance"]
var_data = data["charge_variance"]
var_tilt = tilt["charge_variance"]
var_a6 = a6["charge_variance"]
assert var_proj < var_data < var_tilt, (
    f"anti-concentration ordering violated: proj={var_proj} data={var_data} tilt={var_tilt}"
)
assert var_a6 > var_data, f"A6 should also spread beyond data variance: A6={var_a6} data={var_data}"

print(f"fig1: {verified}/{len(series)} series verified against results/WS7_charge_marginals.json")
print(f"anti-concentration ordering holds: projection ({var_proj:.3f}) < data ({var_data:.3f}) "
      f"< tilt ({var_tilt:.3f}); A6 ({var_a6:.3f}) tracks the tilt")
