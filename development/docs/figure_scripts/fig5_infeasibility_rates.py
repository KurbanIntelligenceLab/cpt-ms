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

with open(REPO / "results" / "infeasibility_correlation.json") as fh:
    artifact = json.load(fh)
payload = artifact["payload"]
rows = sorted(payload["per_target"], key=lambda r: r["target"])

targets = np.array([r["target"] for r in rows])
infeas_final = np.array([r["infeas_final_mean"] for r in rows])
error = np.array([r["abs_error_vs_target"] for r in rows])
corr = payload["correlations"]["abs_error_vs_infeas_final"]

fig, ax = plt.subplots(figsize=(3.4, 3.0))

ax.scatter(infeas_final, error, s=26, color=VIOLET, zorder=3, edgecolors=INK, linewidths=0.4)

offsets = {
    1.5: (5, 12), 2.0: (5, -11), 3.0: (7, 1),
    5.0: (6, -10), 6.317: (-8, 18), 8.0: (7, -2),
    10.0: (7, 4), 13.0: (7, 4), 16.0: (7, 4), 20.0: (7, 4),
}
for r, x, y in zip(rows, infeas_final, error):
    dx, dy = offsets.get(r["target"], (6, 4))
    ax.annotate(f"{r['target']:g}", (x, y), textcoords="offset points",
                xytext=(dx, dy), fontsize=5.6, color=SLATE)

order = np.argsort(infeas_final)
z = np.polyfit(infeas_final, error, 1)
xs = np.linspace(infeas_final.min(), infeas_final.max(), 50)
ax.plot(xs, np.polyval(z, xs), color=SLATE, lw=1.1, ls='--', zorder=1)

ax.set_xlabel("post-training infeasibility fraction")
ax.set_ylabel("absolute density error (g/cm\u00b3)")
ax.margins(0.08)

ax.text(0.03, 0.97,
        f"Spearman {corr['spearman_r']:.2f} (n={corr['n']})\nPearson {corr['pearson_r']:.2f}",
        transform=ax.transAxes, ha='left', va='top', fontsize=6.2, color=INK)

fig.tight_layout()
fig.savefig(REPO / "development" / "figures" / "fig5_infeasibility_rates.png", dpi=300)
fig.savefig(REPO / "development" / "docs" / "template" / "figures" / "fig5_infeasibility_rates.pdf")

scatter = ax.collections[0]
assert scatter.get_offsets().shape[0] == len(rows), "fig5 scatter count mismatch"
plotted_x = scatter.get_offsets()[:, 0]
plotted_y = scatter.get_offsets()[:, 1]
assert np.allclose(np.sort(plotted_x), np.sort(infeas_final)), "fig5 x mismatch"
assert np.allclose(np.sort(plotted_y), np.sort(error)), "fig5 y mismatch"

r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
overlaps = [(a, b) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
assert overlaps == [], f"fig5 label overlap: {overlaps}"

print("fig5: scatter and correlation annotation verified against results/infeasibility_correlation.json")
