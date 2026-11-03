"""Pilot: entanglers with the same metric as the CNOT ladder (n = 4, L = 3). Do they differ in kernel or accuracy?"""
import sys, pickle, json, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bac
from paths import ROOT as P, DATA; from lin import *
warnings.filterwarnings('ignore')
NAMES = ['diabetes', 'ionosphere', 'sonar', 'wdbc', 'vehicle', 'ilpd', 'heart', 'pc1', 'balance', 'phoneme']
n, L = 4, 3; CS = [.05, .1, .2, .4, .8, 1.6]; CG = [.1, 1, 10, 100]
Gs, cls = pickle.load(open(P + '/results/classes.pkl', 'rb'))
key = tuple(np.round(metric(perm_of(ladder(n)), n, L) * 4).astype(int).ravel())
members = cls[L][key]; rng = np.random.default_rng(1)
idx = [i for i in members if (Gs[i] == ladder(n)).all()] + list(rng.choice([i for i in members if not (Gs[i] == ladder(n)).all()], 11, replace=False))
G = metric(perm_of(ladder(n)), n, L)

def tune(Ks, y, tr):
    """Ks: list over scales of full kernel matrices; pick (scale, C) by 3-fold CV on balanced accuracy."""
    best = (-1, 0, 0)
    for s, K in enumerate(Ks):
        for C in CG:
            sc = []
            for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, y[tr]):
                m = SVC(kernel='precomputed', C=C, class_weight='balanced').fit(K[np.ix_(tr[a], tr[a])], y[tr[a]])
                sc.append(bac(y[tr[b]], m.predict(K[np.ix_(tr[b], tr[a])])))
            if np.mean(sc) > best[0] + 1e-12: best = (np.mean(sc), s, C)
    return best

def job(name, seed):
    d = np.load(f'{DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    sc = StandardScaler().fit(X[tr]); Z = np.nan_to_num(sc.transform(X))
    pca = PCA(n, random_state=0).fit(Z[tr]); F = pca.transform(Z); F = StandardScaler().fit(F[tr]).transform(F)
    res = {}
    def evaluate(Ks):
        cv, s, C = tune(Ks, y, tr)
        m = SVC(kernel='precomputed', C=C, class_weight='balanced').fit(Ks[s][np.ix_(tr, tr)], y[tr])
        return dict(cv=cv, test=bac(y[te], m.predict(Ks[s][np.ix_(te, tr)])), c=CS[s])
    res['surr'] = evaluate([surrogate(c * F, c * F, G) for c in CS])
    kq = {}
    for i in idx:
        p = perm_of(Gs[i]); Ks = []
        for c in CS:
            S = states(c * F, L, p); Ks.append(fidelity(S, S))
        kq[i] = Ks; res[int(i)] = evaluate(Ks)
    # kernel spread within the class at each scale: mean |K_i - K_ladder| over off-diagonal entries
    off = ~np.eye(len(y), dtype=bool)
    res['spread'] = [float(np.mean([np.abs(kq[i][s] - kq[idx[0]][s])[off].mean() for i in idx[1:]])) for s in range(len(CS))]
    res['gap'] = [float(np.abs(kq[idx[0]][s] - surrogate(c * F, c * F, G))[off].mean()) for s, c in enumerate(CS)]
    return name, seed, res

out = Parallel(n_jobs=30)(delayed(job)(nm, s) for nm in NAMES for s in range(5))
json.dump(dict(idx=[int(i) for i in idx], out=out), open(P + '/results/pilot.json', 'w'))
print('done')
