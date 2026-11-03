"""H1: consistent member effect (within-split label permutation, split-half replication). H2: CV selection among K
members vs among K perturbed surrogates. All test scores in points of balanced accuracy."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from paths import ROOT; d = json.load(open(ROOT + '/results/pilot2.json')); lad = f"m{d['lad']}"; M = [f'm{i}' for i in d['mem']]; S = [f'sv{j}' for j in range(len(M))]
rng = np.random.default_rng(0); R = 2000
def effect(T):                                             # T: splits x members; variance of member means after split centring
    return np.var((T - T.mean(1, keepdims=True)).mean(0))
rows, pooled_obs, pooled_null, sh = [], 0, np.zeros(R), []
for name in sorted({o[0] for o in d['out']}):
    outs = [o[2] for o in d['out'] if o[0] == name]
    T = 100 * np.array([[r[m]['test'] for m in M] for r in outs]); Ts = 100 * np.array([[r[m]['test'] for m in S] for r in outs])
    obs = effect(T); null = np.array([effect(np.array([rng.permutation(t) for t in T])) for _ in range(R)])
    pooled_obs += obs; pooled_null += null
    a, b = T[::2].mean(0), T[1::2].mean(0); sh.append(spearmanr(a, b)[0])
    sel = lambda keys: np.mean([100 * r[max(keys, key=lambda k: r[k]['cv'])]['test'] for r in outs])
    rows.append(dict(data=name, p_member=(null >= obs).mean(), surr=np.mean([100 * r['surr']['test'] for r in outs]),
                     ladder=np.mean([100 * r[lad]['test'] for r in outs]), member_mean=T.mean(), sel_member=sel(M + [lad]),
                     sel_surr=sel(S + ['surr']), surr_family_effect_p=(np.array([effect(np.array([rng.permutation(t) for t in Ts])) for _ in range(500)]) >= effect(Ts)).mean(),
                     dist_member=np.mean([r['dist_member'] for r in outs]), dist_surr=np.mean([r['dist_surr'] for r in outs])))
df = pd.DataFrame(rows).set_index("data"); pd.set_option("display.width", 200); print(df.round(2))
print('\nH1 pooled member-effect p =', (pooled_null >= pooled_obs).mean(), '| split-half Spearman per data set:', np.round(sh, 2), 'mean', np.round(np.mean(sh), 2))
dd = df.sel_member - df.sel_surr
print('H2 selected member - selected surrogate: mean %.2f, data sets won %d/%d, Wilcoxon p %.3f' % (dd.mean(), (dd > 0).sum(), len(dd), wilcoxon(dd).pvalue))
print('ladder - surrogate: %.2f | selected member - surrogate: %.2f' % ((df.ladder - df.surr).mean(), (df.sel_member - df.surr).mean()))
