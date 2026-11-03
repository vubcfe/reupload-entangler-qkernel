"""Analyses for PREREG_GENERAL.md and PREREG_SHOTS.md. Points of balanced accuracy, 90% bootstrap over data sets."""
import sys, glob, json, os, numpy as np
from scipy.stats import spearmanr
from paths import ROOT as P; mode = sys.argv[1]; rng = np.random.default_rng(0)
recs = [(os.path.basename(f)[:-5].rsplit('_', 1)[0], json.load(open(f))) for f in glob.glob(f'{P}/results/{mode}/*.json')]
def boot(v):
    ks = list(v); arr = [np.array(v[k]) for k in ks]
    bs = [np.mean(np.concatenate([arr[i] for i in rng.integers(len(ks), size=len(ks))])) for _ in range(10000)]
    return np.mean(np.concatenate(arr)), np.percentile(bs, [5, 95])
def line(tag, v, rule):
    m, ci = boot(v); dd = np.array([np.mean(x) for x in v.values()])
    return f'  {tag}: {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] supported={rule(m, ci)} positive {int((dd > 0).sum())}/{len(dd)}'
if mode == 'general':
    for st in ['n4L2', 'n4L4', 'n4L6', 'n6L3']:
        R = [(d, r) for d, r in recs if f'{st}_lad_u' in r]; data = sorted({d for d, _ in R})
        S = sorted({k.split('_')[1] for k in R[0][1] if k.startswith(st + '_s')}); F = sorted({k.split('_')[1] for k in R[0][1] if k.startswith(st + '_f')})
        g = lambda r, k: r[f'{st}_{k}']['test']; per = lambda fn: {d: [100 * fn(r) for dd, r in R if dd == d] for d in data}
        print(f'{st}: {len(data)} data sets, {len(R)} jobs')
        print(line('G1 singular - ladder', per(lambda r: np.mean([g(r, s + '_u') for s in S]) - g(r, 'lad_u')), lambda m, ci: ci[1] < 0))
        print(line('G2 singular geometric - uniform', per(lambda r: np.mean([g(r, s + '_g') - g(r, s + '_u') for s in S])), lambda m, ci: ci[0] > 0))
        print(line('   full-rank - ladder', per(lambda r: np.mean([g(r, f + '_u') for f in F]) - g(r, 'lad_u')), lambda m, ci: None))
        print(line('   singular geometric - ladder', per(lambda r: np.mean([g(r, s + '_g') for s in S]) - g(r, 'lad_u')), lambda m, ci: None))
        arms = ['lad'] + S + F; rho = [R[0][1][f'{st}_{a}_u']['rho'] for a in arms]
        eff = [100 * np.mean([g(r, a + '_u') - g(r, 'lad_u') for _, r in R]) for a in arms]
        print('   Spearman(rho, effect) over uniform arms: %+.2f' % spearmanr(rho, eff)[0])
else:
    data = sorted({d for d, _ in recs}); r0 = recs[0][1]; S = sorted({k.split('_')[0] for k in r0 if k[0] == 's'})
    print(f'shots: {len(recs)} jobs, {len(data)} data sets')
    for Sh in (0, 1000, 100):
        g = lambda r, k: r[f'{k}_S{Sh}']['test']; per = lambda fn: {d: [100 * fn(r) for dd, r in recs if dd == d] for d in data}
        print(f'S = {Sh if Sh else "exact"}:')
        print(line('S1 singular uniform - ladder', per(lambda r: np.mean([g(r, s + '_u') for s in S]) - g(r, 'lad_u')), lambda m, ci: ci[1] < 0))
        print(line('S2 singular geometric - uniform', per(lambda r: np.mean([g(r, s + '_g') - g(r, s + '_u') for s in S])), lambda m, ci: ci[0] > 0))
        print('   mean accuracy: ladder %.2f, singular uniform %.2f, singular geometric %.2f' % tuple(
            100 * np.mean([np.mean([g(r, k) if k == 'lad_u' else np.mean([g(r, s + k) for s in S]) for _, r in recs])]) for k in ('lad_u', '_u', '_g')))
