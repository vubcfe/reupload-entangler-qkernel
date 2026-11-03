"""Figures for the journal paper, from the remedy run (all 23 data sets, seeds 100-109)."""
import sys, glob, json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paths import ROOT as P
import main as M, confirm as CF
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6, 'pdf.fonttype': 42})
INK, MUTED, GRID, BLUE, ORANGE = '#222222', '#6b6b6b', '#e4e4e4', '#2a78d6', '#eb6834'
recs = [(os.path.basename(f)[:-5].rsplit('_', 2)[0], json.load(open(f))) for f in glob.glob(P + '/results/remedy/*.json')]
data = sorted({d for d, _ in recs})
S = [int(j) for j in CF.S_ARMS]; F = [int(j) for j in CF.F_ARMS[:10]]
def rho(j, w=None):
    G = M.metric(M.perm_of(M.Gs[M.cls[3][M.KEYS[j]][0]]), 4, 3, w); return np.linalg.eigvalsh(G)[0] * 4 / np.trace(G)
eff = lambda k: 100 * np.mean([r[k]['test'] - r['lad_u']['test'] for _, r in recs])
geo = np.array([1, .5, .25])

# Fig 1: arm effect against relative smallest eigenvalue of G
fig, a = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
a.axhline(0, color=MUTED, lw=.6, ls='--', zorder=1)
a.scatter([rho(j) for j in S], [eff(f's{j}_u') for j in S], s=22, color=BLUE, edgecolor='white', lw=.6, zorder=3, label='singular, uniform $w$')
a.scatter([rho(j) for j in F], [eff(f'f{j}_u') for j in F], s=22, marker='s', color=BLUE, alpha=.55, edgecolor='white', lw=.6, zorder=3, label='full rank, uniform $w$')
a.scatter([rho(j, geo) for j in S], [eff(f's{j}_g') for j in S], s=22, marker='^', color=ORANGE, edgecolor='white', lw=.6, zorder=3, label='singular, geometric $w$')
a.scatter([0.6], [0], s=40, marker='*', color=INK, zorder=4, label='CNOT ladder')
a.set_xlabel(r'$\lambda_{\min}(G)\,n/\mathrm{tr}\,G$'); a.set_ylabel('arm $-$ ladder (points)')
a.grid(axis='y', color=GRID, lw=.5); a.legend(frameon=False, fontsize=6, loc='lower right', labelcolor=INK)
fig.savefig(P + '/figures/fig_conditioning.pdf'); plt.close(fig)

# Fig 2: per data set, singular entanglers with uniform vs geometric scales (relative to the ladder)
u = {d: 100 * np.mean([np.mean([r[f's{j}_u']['test'] for j in S]) - r['lad_u']['test'] for dd, r in recs if dd == d]) for d in data}
g = {d: 100 * np.mean([np.mean([r[f's{j}_g']['test'] for j in S]) - r['lad_u']['test'] for dd, r in recs if dd == d]) for d in data}
order = sorted(data, key=lambda d: u[d])
fig, a = plt.subplots(figsize=(3.4, 3.3), constrained_layout=True)
yy = np.arange(len(order))
a.axvline(0, color=MUTED, lw=.6, ls='--', zorder=1)
for i, d in enumerate(order): a.plot([u[d], g[d]], [i, i], color=GRID, lw=1.2, zorder=1)
a.scatter([u[d] for d in order], yy, s=18, color=BLUE, edgecolor='white', lw=.5, zorder=3, label='uniform $w$')
a.scatter([g[d] for d in order], yy, s=18, marker='^', color=ORANGE, edgecolor='white', lw=.5, zorder=3, label='geometric $w=(1,1/2,1/4)$')
a.set_yticks(yy); a.set_yticklabels(order, fontsize=6); a.set_xlabel('singular entanglers $-$ ladder (points)')
a.set_xscale('symlog', linthresh=2); a.grid(axis='x', color=GRID, lw=.5)
T = [-10, -5, -2, -1, 0, 1, 2]; a.set_xticks(T); a.set_xticklabels([str(t) for t in T]); a.minorticks_off()
a.legend(frameon=False, fontsize=6, loc='upper left', labelcolor=INK)
fig.savefig(P + '/figures/fig_remedy.pdf'); plt.close(fig)
print('ok')
