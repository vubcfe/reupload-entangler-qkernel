"""Analyses for PREREG_CONFIRM.md (mode confirm) and PREREG_REMEDY.md (mode remedy). Points of balanced accuracy."""
import sys, glob, json, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, binomtest
from paths import ROOT as P; mode = sys.argv[1]; rng = np.random.default_rng(0)
recs = []
for f in glob.glob(f'{P}/results/{mode}/*.json'):
    nm, s, p = os.path.basename(f)[:-5].rsplit('_', 2); recs.append((nm, p, int(s), json.load(open(f))))
data = sorted({r[0] for r in recs}); print(f'{mode}: {len(recs)} jobs, {len(data)} data sets')

def per_data(fn):
    """fn(record) -> value in points; averaged over splits and pipelines within each data set."""
    return {d: [100 * fn(r[3]) for r in recs if r[0] == d] for d in data}

def boot(v, B=10000):
    ks = list(v); arr = [np.array(v[k]) for k in ks]
    bs = [np.mean(np.concatenate([arr[i] for i in rng.integers(len(ks), size=len(ks))])) for _ in range(B)]
    return np.mean(np.concatenate(arr)), np.percentile(bs, [5, 95])

def show(tag, v, rule):
    m, ci = boot(v); dd = np.array([np.mean(x) for x in v.values()])
    print(f'{tag}: mean {m:+.2f}, 90% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], supported = {rule(m, ci)}, '
          f'Wilcoxon p = {wilcoxon(dd).pvalue:.4f}, positive in {(dd > 0).sum()}/{len(dd)}'); return dd

r0 = recs[0][3]
if mode == 'confirm':
    S = sorted(k for k in r0 if k[0] == 's'); F = sorted(k for k in r0 if k[0] == 'f')
    s = show('C1 singular - ladder', per_data(lambda r: np.mean([r[k]['test'] for k in S]) - r['lad']['test']), lambda m, ci: ci[1] < 0 and m < -1)
    f = show('C2 full-rank - ladder', per_data(lambda r: np.mean([r[k]['test'] for k in F]) - r['lad']['test']), lambda m, ci: ci[0] > -.5 and ci[1] < .5)
    print(f'C3 singular - full-rank: mean {np.mean(s - f):+.2f}, one-sided Wilcoxon p = {wilcoxon(s - f, alternative="less").pvalue:.4f}')
    rho = []
    for d in data:
        dv, lv = [], []
        for r in [r for r in recs if r[0] == d]:
            for k in S: dv.append(r[3]['rbf_full'] - r[3][k]['rbf_proj']); lv.append(r[3][k]['test'] - r[3]['lad']['test'])
        rho.append(spearmanr(dv, lv)[0])
    rho = np.array(rho); k = int((rho < 0).sum())
    print(f'C4 mechanism: within-data-set Spearman(d, loss) {np.round(rho, 2)}; negative in {k}/{len(rho)}, '
          f'sign test p = {binomtest(k, len(rho), alternative="greater").pvalue:.4f}')
    allv = [(r[3]['rbf_full'] - r[3][k]['rbf_proj'], r[3][k]['test'] - r[3]['lad']['test']) for r in recs for k in S]
    print('   pooled Spearman %.2f' % spearmanr(*zip(*allv))[0])
else:
    S = sorted({k[:-2] for k in r0 if k[0] == 's'}); F = sorted({k[:-2] for k in r0 if k[0] == 'f'})
    show('R1 singular: geometric - uniform', per_data(lambda r: np.mean([r[k + '_g']['test'] - r[k + '_u']['test'] for k in S])), lambda m, ci: m > 1 and ci[0] > 0)
    show('R2 singular geometric - uniform ladder', per_data(lambda r: np.mean([r[k + '_g']['test'] for k in S]) - r['lad_u']['test']), lambda m, ci: ci[0] > -1 and ci[1] < 1)
    show('R3 full-rank: geometric - uniform', per_data(lambda r: np.mean([r[k + '_g']['test'] - r[k + '_u']['test'] for k in F])), lambda m, ci: ci[0] > -.5 and ci[1] < .5)
    show('R4 ladder: geometric - uniform', per_data(lambda r: r['lad_g']['test'] - r['lad_u']['test']), lambda m, ci: ci[0] > -.5 and ci[1] < .5)
    show('(reported) singular uniform - uniform ladder', per_data(lambda r: np.mean([r[k + '_u']['test'] for k in S]) - r['lad_u']['test']), lambda m, ci: None)
