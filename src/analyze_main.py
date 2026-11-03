"""Preregistered analysis (PREREGISTRATION.md, revision 2). Points of balanced accuracy; cell = data set x pipeline."""
import sys, glob, json, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, binomtest
from paths import ROOT as P; rng = np.random.default_rng(0)
recs = []
for f in glob.glob(P + '/results/main/*.json'):
    name, seed, pipe = os.path.basename(f)[:-5].rsplit('_', 2); recs.append((name, int(seed), pipe, json.load(open(f))))
cells = sorted({(r[0], r[2]) for r in recs}); data = sorted({c[0] for c in cells})
print(f'{len(recs)} jobs, {len(cells)} cells, {len(data)} data sets')
r0 = recs[0][3]; W = sorted(k for k in r0 if k.startswith('w') and k[1:].isdigit())
B = sorted(k for k in r0 if k.startswith('b') and k[1:].isdigit()); SV = [f'sv{j}' for j in range(30)]

def mat(cell, keys):
    rs = sorted([r for r in recs if (r[0], r[2]) == cell], key=lambda r: r[1])
    return 100 * np.array([[r[3][k]['test'] for k in keys] for r in rs]), rs

def effect(T): return np.var((T - T.mean(1, keepdims=True)).mean(0))
def member_sd(T): return np.sqrt(effect(T))

def perm_test(keys, R=5000):
    obs, null, sds = 0, np.zeros(R), {}
    for c in cells:
        T, _ = mat(c, keys); obs += effect(T); sds[c] = member_sd(T)
        null += np.array([effect(np.array([rng.permutation(t) for t in T])) for _ in range(R)])
    return (null >= obs).mean(), sds

def boot_mean(vals_by_data, B_=10000, q=(5, 95)):
    ks = list(vals_by_data); arr = [np.array(vals_by_data[k]) for k in ks]
    bs = [np.mean(np.concatenate([arr[i] for i in rng.integers(len(ks), size=len(ks))])) for _ in range(B_)]
    return np.mean(np.concatenate(arr)), np.percentile(bs, q)

def by_data(cellvals):
    d = {}
    for (nm, p), v in cellvals.items(): d.setdefault(nm, []).append(v)
    return d

out = {}
for tag, keys in [('H1 within class', ['lad'] + W), ('H2 between classes', ['lad'] + B)]:
    p, sds = perm_test(keys); m, ci = boot_mean(by_data(sds))
    print(f'{tag}: permutation p = {p:.3f}; member SD mean {m:.2f}, 90% CI [{ci[0]:.2f}, {ci[1]:.2f}]; '
          f'supported = {p > .05 and ci[1] < .5}')
def diff(fa, fb):
    v = {}
    for c in cells:
        rs = [r[3] for r in recs if (r[0], r[2]) == c]; v[c] = 100 * np.mean([fa(r) - fb(r) for r in rs])
    return v
def report(tag, v, margin=1):
    m, ci = boot_mean(by_data(v)); dd = pd.Series({k: np.mean(x) for k, x in by_data(v).items()})
    print(f'{tag}: mean {m:+.2f}, 90% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], equivalent(±{margin}) = {ci[0] > -margin and ci[1] < margin}, '
          f'Wilcoxon p = {wilcoxon(dd).pvalue:.3f}, positive in {(dd > 0).sum()}/{len(dd)} data sets'); return dd
report('H3 design - ladder', diff(lambda r: r['des']['test'], lambda r: r['lad']['test']))
sel = lambda r, keys, suf: r[max(keys, key=lambda k: r[k + suf]['cv']) + suf]['test']
fix = diff(lambda r: sel(r, ['lad'] + B, '_fix'), lambda r: r['lad_fix']['test'])
bw = diff(lambda r: sel(r, ['lad'] + B, ''), lambda r: r['lad']['test'])
report('H4 search gain at fixed c=1 (D_fix)', fix); dbw = report('H4 search gain with matched bandwidth (D_bw)', bw)
dd = pd.Series({k: np.mean(x) for k, x in by_data({c: fix[c] - bw[c] for c in cells}).items()})
print(f'H4 D_fix - D_bw: mean {dd.mean():+.2f}, one-sided Wilcoxon p = {wilcoxon(dd, alternative="greater").pvalue:.3f}')
report('best fixed-scale search - best matched search', diff(lambda r: sel(r, ['lad'] + B, '_fix'), lambda r: sel(r, ['lad'] + B, '')))
rho = []
for c in cells:
    T, rs = mat(c, W); dist = np.array([[r[3][k]['dist'] for k in W] for r in rs]).mean(0); rho.append(spearmanr(dist, T.mean(0))[0])
rho = np.array(rho); print(f'Secondary: Spearman(dist, acc) median {np.nanmedian(rho):+.2f}, negative in {(rho < 0).sum()}/{len(rho)}, '
                           f'sign test p = {binomtest(int((rho < 0).sum()), int((~np.isnan(rho)).sum()), alternative="greater").pvalue:.3f}')
report('ladder - surrogate', diff(lambda r: r['lad']['test'], lambda r: r['surr']['test']))
report('selected surrogate family - selected T1 member', diff(lambda r: sel(r, ['surr'] + SV, ''), lambda r: sel(r, ['lad'] + W, '')))
