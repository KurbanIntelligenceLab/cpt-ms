"""Figure 4 -- infeasibility at convergence vs. requested target, with repair factor.

Reads results/fig_infeasibility_vs_target.json exclusively. No number in
this script is typed by hand. Repair factor = ratio of seed-averaged means
(infeas_init_mean / infeas_final_mean) per target -- see the artifact's own
`repair_factor_convention` field.
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

with open(REPO / "results" / "fig_infeasibility_vs_target.json") as fh:
    artifact = json.load(fh)
payload = artifact["payload"]
rows = sorted(payload["per_target"], key=lambda r: r["target"])

targets = np.array([r["target"] for r in rows])
init = np.array([r["infeas_init_mean"] for r in rows])
final = np.array([r["infeas_final_mean"] for r in rows])
repair = np.array([r["repair_factor_ratio_of_means"] for r in rows])

fig, ax = plt.subplots(figsize=(3.4, 3.0))

ax.plot(targets, init, color=SLATE, lw=1.6, ls='--', marker='o', markersize=4,
        label='initial (pre-training)', zorder=2)
ax.plot(targets, final, color=VIOLET, lw=1.8, ls='-', marker='o', markersize=4,
        label='final (deployed, post-training)', zorder=3)

ax2 = ax.twinx()
ax2.plot(targets, repair, color=LILAC, lw=1.2, ls=':', marker='^', markersize=4,
         label='repair factor (init / final)', zorder=1)
ax2.set_ylabel("repair factor (\u00d7)", color=LILAC)
ax2.tick_params(axis='y', colors=LILAC)
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color(LILAC)

# Headline annotation at the weakest steering target.
weakest = rows[0]
ax.annotate(f"{weakest['repair_factor_ratio_of_means']:.1f}\u00d7 repair\nat target {weakest['target']:g}",
            xy=(weakest['target'], weakest['infeas_init_mean']),
            xytext=(weakest['target'] + 0.16 * (targets.max() - targets.min()),
                    weakest['infeas_init_mean'] + 0.02),
            fontsize=6.2, color=SLATE,
            arrowprops=dict(arrowstyle='-', color=SLATE, lw=0.8))

ax.set_xlabel("requested density target (g/cm\u00b3)")
ax.set_ylabel("per-step infeasibility fraction")
ax.margins(y=0.08)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=6.0,
          handlelength=1.6, frameon=False)

fig.tight_layout()
fig.savefig(REPO / "development" / "figures" / "fig4_infeasibility_vs_target.png", dpi=300)
fig.savefig(REPO / "development" / "docs" / "template" / "figures" / "fig4_infeasibility_vs_target.pdf")

# --- Verification
init_line = [ln for ln in ax.get_lines() if ln.get_label() == 'initial (pre-training)'][0]
final_line = [ln for ln in ax.get_lines() if ln.get_label() == 'final (deployed, post-training)'][0]
repair_line = [ln for ln in ax2.get_lines() if ln.get_label() == 'repair factor (init / final)'][0]

assert np.allclose(init_line.get_xdata(), targets) and np.allclose(init_line.get_ydata(), init), "fig4 init mismatch"
assert np.allclose(final_line.get_xdata(), targets) and np.allclose(final_line.get_ydata(), final), "fig4 final mismatch"
assert np.allclose(repair_line.get_xdata(), targets) and np.allclose(repair_line.get_ydata(), repair), "fig4 repair mismatch"

print("fig4: 3/3 series verified against results/fig_infeasibility_vs_target.json")
