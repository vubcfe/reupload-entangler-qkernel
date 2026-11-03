"""Data overview figure: size and dimension of the 23 data sets, and variance kept by 4 and 6 PCA components."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from paths import ROOT, DATA
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .6, 'pdf.fonttype': 42})
INK, MUTED, GRID, BLUE, ORANGE = '#222222', '#6b6b6b', '#e4e4e4', '#2a78d6', '#eb6834'
SET_A = {'banknote', 'blood', 'climate', 'iris', 'kc1', 'magic', 'optdigits', 'qsar', 'segment', 'spambase', 'steel', 'wilt', 'wine'}
meta = json.load(open(DATA + '/meta.json')); names = sorted(meta)
var = {}
for nm in names:
    X = np.nan_to_num(StandardScaler().fit_transform(np.load(f'{DATA}/{nm}.npz')['X']))
    r = PCA(random_state=0).fit(X).explained_variance_ratio_; var[nm] = (r[:4].sum(), r[:6].sum() if len(r) >= 6 else np.nan)
order = sorted(names, key=lambda k: (meta[k]['d'], k)); y = np.arange(len(order))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, constrained_layout=True, gridspec_kw={'width_ratios': [1, 1.2]})
a = ax[0]
for i, nm in enumerate(order):
    m = meta[nm]; a.scatter(m['d'], i, s=14 + 5 * m['k'], color=BLUE if nm in SET_A else ORANGE, edgecolor='white', lw=.6, zorder=3)
    a.text(m['d'] * 1.12, i, f"N={m['n']}, K={m['k']}", fontsize=5.5, color=MUTED, va='center')
a.set_xscale('log'); a.set_xlim(3, 400); a.axvline(4, color=MUTED, lw=.6, ls='--'); a.axvline(6, color=MUTED, lw=.6, ls=':')
a.set_xlabel('number of features $d$ (log scale)'); a.set_yticks(y); a.set_yticklabels(order, fontsize=6)
a.scatter([], [], color=BLUE, s=20, label='set A (13)'); a.scatter([], [], color=ORANGE, s=20, label='set B (10)')
a.legend(frameon=False, fontsize=6, loc='lower right', labelcolor=INK); a.grid(axis='x', color=GRID, lw=.5)
a.set_title('(a) features $d$, examples $N$, classes $K$', fontsize=8)
b = ax[1]
b.barh(y + .18, [var[k][0] for k in order], height=.36, color=BLUE, label='4 components')
b.barh(y - .18, [var[k][1] for k in order], height=.36, color=ORANGE, label='6 components')
b.set_xlim(0, 1.02); b.set_xlabel('fraction of variance kept by PCA')
b.grid(axis='x', color=GRID, lw=.5); b.legend(frameon=False, fontsize=6, loc='upper right', labelcolor=INK); b.set_title('(b) variance kept after reduction', fontsize=8)
fig.savefig(ROOT + '/figures/fig_data.pdf'); print({k: tuple(round(x, 2) for x in v) for k, v in var.items()})
