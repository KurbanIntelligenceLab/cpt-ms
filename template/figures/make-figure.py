"""Generates the demo figure in the KIL palette + Montserrat. Replace with your own."""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
import glob

for f in glob.glob('/usr/share/texlive/texmf-dist/fonts/opentype/public/montserrat/Montserrat-*.otf'):
    fm.fontManager.addfont(f)

INK, VIOLET, LILAC, SLATE, LINE = '#0D1526', '#6C5CF6', '#B49BFB', '#5C6C87', '#E3DFF8'
mpl.rcParams.update({
    'font.family': 'Montserrat', 'font.size': 8.5,
    'axes.edgecolor': LINE, 'axes.labelcolor': INK, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': SLATE, 'ytick.color': SLATE, 'text.color': INK,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.frameon': False, 'figure.dpi': 200,
})

fig, (a, b) = plt.subplots(1, 2, figsize=(6.0, 2.15), gridspec_kw={'wspace': 0.30})

rng = np.random.default_rng(7); e = np.arange(1, 61)
for lbl, col, k, ls in [('KIL-Net (ours)', VIOLET, 0.055, '-'),
                        ('Baseline B', LILAC, 0.038, '-'),
                        ('Baseline A', SLATE, 0.028, '--')]:
    y = 0.28 + 0.66 * (1 - np.exp(-k * e)) + rng.normal(0, .006, e.size)
    a.plot(e, y, color=col, lw=1.7, ls=ls, label=lbl)
    a.fill_between(e, y - .022, y + .022, color=col, alpha=.10, lw=0)
a.set_xlabel('Training epoch'); a.set_ylabel('Validation accuracy')
a.set_ylim(.25, 1.0); a.legend(loc='lower right', fontsize=7)
a.set_title('a', loc='left', fontweight='bold', fontsize=10, color=INK, pad=8)

ds = ['QM9', 'MD17', 'OC20', 'MatBench']
x = np.arange(len(ds)); w = .27
for i, (lbl, col, v) in enumerate([('Baseline A', LILAC, [.71, .68, .63, .74]),
                                   ('Baseline B', SLATE, [.76, .74, .70, .79]),
                                   ('KIL-Net (ours)', VIOLET, [.88, .85, .81, .90])]):
    b.bar(x + (i - 1) * w, v, w * .88, color=col, label=lbl, zorder=3)
b.set_xticks(x); b.set_xticklabels(ds); b.set_ylabel('Macro-F1'); b.set_ylim(.5, 1.0)
b.grid(axis='y', color=LINE, lw=.8, zorder=0); b.legend(fontsize=7, ncol=1, loc='upper left')
b.set_title('b', loc='left', fontweight='bold', fontsize=10, color=INK, pad=8)

fig.savefig('example-figure.pdf', bbox_inches='tight', transparent=True)
print('wrote figures/example-figure.pdf')


# --- standalone panels for the subfigure example in the appendix ---------------
# Same brand rules, exported one axes per file. Replace with real panels.
_rng = np.random.default_rng(7)
_specs = [('KIL-Net (ours)', VIOLET, 0.055, '-'),
          ('Baseline B',     LILAC,  0.045, '-'),
          ('Baseline A',     SLATE,  0.038, '--')]
for name, which in [("example-panel-a", "curves"), ("example-panel-b", "bars")]:
    fg, ax1 = plt.subplots(figsize=(3.1, 2.5))
    if which == "curves":
        for lbl, col, k, ls in _specs:
            y = 1 - 0.72 * np.exp(-k * e) + _rng.normal(0, .008, e.size)
            ax1.plot(e, y, ls, color=col, lw=2 if col == VIOLET else 1.5, label=lbl)
        ax1.set_xlabel("Training epoch"); ax1.set_ylabel("Validation accuracy")
        ax1.legend(frameon=False, fontsize=7, loc="lower right")
    else:
        vals = [("Baseline A", LILAC, [.71, .68, .63, .74]),
                ("Baseline B", SLATE, [.76, .74, .70, .79]),
                ("KIL-Net (ours)", VIOLET, [.88, .85, .81, .90])]
        for i, (lbl, col, v) in enumerate(vals):
            ax1.bar(x + (i - 1) * w, v, w, color=col, label=lbl)
        ax1.set_xticks(x, ds); ax1.set_ylabel("Macro-F1"); ax1.set_ylim(0.5, 1.0)
        ax1.legend(frameon=False, fontsize=7)
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
    fg.tight_layout()
    fg.savefig(f"{name}.pdf", transparent=True)
    plt.close(fg)
print("panels written")
