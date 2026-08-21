from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

REPO = Path(__file__).resolve().parents[3]

for f in glob.glob('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/montserrat/Montserrat-*.otf'):
    fm.fontManager.addfont(f)

apply_figure_style(font='Montserrat')

INK, VIOLET, LILAC, SLATE, LINE = '#0D1526', '#6C5CF6', '#B49BFB', '#5C6C87', '#E3DFF8'
ALARM = '#C0392B'
M2_REALISTIC = '#5C6C87'
M2_COLLINEAR = '#C0392B'

with open(REPO / "results" / "solver_robustness.json") as fh:
    artifact = json.load(fh)
payload = artifact["payload"]

tol_m1 = payload["tolerance_sweep"]["m1"]["rows"]
tol_m2 = payload["tolerance_sweep"]["m2"]["rows"]

tols_m1 = np.array([r["tol"] for r in tol_m1])
grad_err_m1 = np.array([r["grad_max_rel_error_vs_tight_reference"] for r in tol_m1])
resid_m1 = np.array([r["achieved_residual_max"] for r in tol_m1])
wall_m1 = np.array([r["wall_clock_forward_s"] for r in tol_m1]) * 1e3

boundary_m1 = payload["boundary_conditioning"]["m1"]["results"]
lo_m1 = sorted([r for r in boundary_m1 if r["side"] == "lo" and r["converged_fraction"] > 0.99],
               key=lambda r: r["frac_from_boundary"])
hi_m1 = sorted([r for r in boundary_m1 if r["side"] == "hi" and r["converged_fraction"] > 0.99],
               key=lambda r: r["frac_from_boundary"])

boundary_m2r = payload["boundary_conditioning"]["m2_realistic_pair"]["results"]
boundary_m2c = payload["boundary_conditioning"]["m2_near_collinear_pair"]["results"]
lo_m2r = sorted([r for r in boundary_m2r if r["side"] == "lo" and r["frac_from_boundary"] > 0],
                key=lambda r: r["frac_from_boundary"])
lo_m2c = sorted([r for r in boundary_m2c if r["side"] == "lo" and r["frac_from_boundary"] > 0],
                key=lambda r: r["frac_from_boundary"])

fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0))
ax_a, ax_b, ax_c = axes

ax_a.plot(tols_m1, grad_err_m1, color=VIOLET, lw=1.8, marker='o', markersize=3.6,
          label='gradient error vs. tight reference', zorder=3)
ax_a.plot(tols_m1, resid_m1, color=SLATE, lw=1.4, ls='--', marker='s', markersize=3.2,
          label='achieved constraint residual', zorder=2)
ax_a.set_xscale('log')
ax_a.set_yscale('log')
ax_a.invert_xaxis()
ax_a.set_xlabel("root-finding tolerance", labelpad=8)
ax_a.set_ylabel("error (log scale)", labelpad=8)
ax_a.legend(loc='lower left', fontsize=6.0, frameon=False, handlelength=1.6,
            bbox_to_anchor=(0.0, 0.16))
ax_a.margins(x=0.1)
ax_a.set_ylim(grad_err_m1.min() * 0.15, resid_m1.max() * 8)

ax_a2 = ax_a.twinx()
ax_a2.plot(tols_m1, wall_m1, color=LILAC, lw=1.1, ls=':', marker='^', markersize=3.2, zorder=1)
ax_a2.set_ylabel("wall clock (ms)", color=LILAC)
ax_a2.tick_params(axis='y', colors=LILAC)
ax_a2.spines['right'].set_visible(True)
ax_a2.spines['right'].set_color(LILAC)
ax_a2.set_ylim(0, wall_m1.max() * 1.25)

fracs_lo_m1 = np.array([r["frac_from_boundary"] for r in lo_m1])
mult_lo_m1 = np.array([r["multiplier_absmax_over_batch"] for r in lo_m1])
fracs_hi_m1 = np.array([r["frac_from_boundary"] for r in hi_m1])
mult_hi_m1 = np.array([r["multiplier_absmax_over_batch"] for r in hi_m1])

ax_b.plot(fracs_lo_m1, mult_lo_m1, color=VIOLET, lw=1.8, marker='o', markersize=3.6,
          label='target near lower bound', zorder=3)
ax_b.plot(fracs_hi_m1, mult_hi_m1, color=SLATE, lw=1.6, ls='--', marker='s', markersize=3.2,
          label='target near upper bound', zorder=2)
ax_b.set_xscale('log')
ax_b.set_yscale('log')
ax_b.invert_xaxis()
ax_b.set_xlabel("distance from boundary (frac. of range)", labelpad=8)
ax_b.set_ylabel("multiplier magnitude, max over batch")
ax_b.legend(loc='upper right', fontsize=6.0, frameon=False, handlelength=1.6)
ax_b.margins(x=0.06)

fracs_r = np.array([r["frac_from_boundary"] for r in lo_m2r])
cond_r = np.array([r["hessian_metric_max_over_batch"] for r in lo_m2r])
fracs_c = np.array([r["frac_from_boundary"] for r in lo_m2c])
cond_c = np.array([r["hessian_metric_max_over_batch"] for r in lo_m2c])

ax_c.plot(fracs_r, cond_r, color=M2_REALISTIC, lw=1.8, marker='o', markersize=3.6,
          label='mass + valence density (corr. 0.73)', zorder=3)
ax_c.plot(fracs_c, cond_c, color=M2_COLLINEAR, lw=1.8, marker='^', markersize=3.6,
          label='near-collinear constraint pair', zorder=4)
ax_c.set_xscale('log')
ax_c.set_yscale('log')
ax_c.invert_xaxis()
ax_c.set_xlabel("distance from boundary (frac. of range)", labelpad=8)
ax_c.set_ylabel("dual Hessian condition number")
ax_c.legend(loc='upper right', fontsize=6.0, frameon=False, handlelength=1.6,
            bbox_to_anchor=(0.88, 1.0))
ax_c.margins(x=0.06)

for ax, letter in zip(axes, 'abc'):
    panel_letter(ax, letter)

fig.tight_layout(w_pad=1.8)
fig.subplots_adjust(bottom=0.18)
fig.savefig(REPO / "development" / "figures" / "fig6_solver_robustness.png", dpi=300)
fig.savefig(REPO / "development" / "docs" / "template" / "figures" / "fig6_solver_robustness.pdf")
fig.savefig(REPO / "development" / "docs" / "template" / "figures" / "fig6_solver_robustness.png", dpi=300)

assert np.allclose(ax_a.get_lines()[0].get_ydata(), grad_err_m1, equal_nan=True)
assert np.allclose(ax_a.get_lines()[1].get_ydata(), resid_m1, equal_nan=True)
assert np.allclose(ax_b.get_lines()[0].get_ydata(), mult_lo_m1, equal_nan=True)
assert np.allclose(ax_b.get_lines()[1].get_ydata(), mult_hi_m1, equal_nan=True)
assert np.allclose(ax_c.get_lines()[0].get_ydata(), cond_r, equal_nan=True)
assert np.allclose(ax_c.get_lines()[1].get_ydata(), cond_c, equal_nan=True)

r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
overlaps = [(a, b) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
assert overlaps == [], f"{len(overlaps)} overlapping text boxes"

print("fig6: 3 panels verified against results/solver_robustness.json")
