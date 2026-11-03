"""Pilot 2 on the development data sets, with the reviewer's fixes: per-member bandwidth calibration to a label-free
statistic, one member per symmetry orbit, an equal-budget family of perturbed Gaussian surrogates, 10 splits."""
import sys, pickle, json, itertools, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bac
from paths import ROOT as P, DATA; from lin import *
warnings.filterwarnings('ignore')
DEV = ['diabetes', 'ionosphere', 'sonar', 'wdbc', 'vehicle', 'ilpd', 'heart', 'pc1', 'balance', 'phoneme']
n, L, K, SPLITS = 4, 3, 12, 10; CG = [.1, 1, 10, 100]
C0 = np.geomspace(.05, 1.6, 12)                       # surrogate scales that define the target bandwidths
Gs, cls = pickle.load(open(P + '/results/classes.pkl', 'rb'))
G = metric(perm_of(ladder(n)), n, L); mem = cls[L][tuple(np.round(G * 4).astype(int).ravel())]
stab = [np.eye(n, dtype=int)[list(p)] for p in itertools.permutations(range(n))
        if np.allclose(np.eye(n)[list(p)] @ G @ np.eye(n)[list(p)].T, G)]
look = {Gs[i].tobytes(): i for i in mem}
reps = sorted({min(look[(Q @ Gs[i] @ Q.T).tobytes()] for Q in stab) for i in mem})
lad = look[ladder(n).tobytes()]; lad = min(look[(Q @ Gs[lad] @ Q.T).tobytes()] for Q in stab)
rng = np.random.default_rng(2); MEM = list(rng.choice([r for r in reps if r != lad], K, replace=False))
SIG = 0.15                                              # log-scale perturbation of G for the surrogate family (set in calib)
PERT = [np.exp(np.random.default_rng(100 + j).normal(0, SIG, n)) for j in range(K)]

def stat(Kmat):                                         # label-free bandwidth statistic: median off-diagonal -log k
    off = ~np.eye(len(Kmat), dtype=bool); return np.median(-np.log(np.clip(Kmat[off], 1e-300, 1)))

def calibrate(f, F, targets):
    """For kernel family c -> f(c*F), find c with stat = target, by bisection in log c."""
    out = []
    for t in targets:
        lo, hi = np.log(1e-3), np.log(5.)
        for _ in range(22):
            mid = (lo + hi) / 2
            if stat(f(np.exp(mid) * F)) < t: lo = mid
            else: hi = mid
        out.append(np.exp((lo + hi) / 2))
    return out

def tune_eval(Ks, y, tr, te):
    best = (-1, 0, 0)
    for s, Km in enumerate(Ks):
        for C in CG:
            sc = []
            for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, y[tr]):
                m = SVC(kernel='precomputed', C=C, class_weight='balanced').fit(Km[np.ix_(tr[a], tr[a])], y[tr[a]])
                sc.append(bac(y[tr[b]], m.predict(Km[np.ix_(tr[b], tr[a])])))
            if np.mean(sc) > best[0] + 1e-12: best = (np.mean(sc), s, C)
    cv, s, C = best
    m = SVC(kernel='precomputed', C=C, class_weight='balanced').fit(Ks[s][np.ix_(tr, tr)], y[tr])
    return dict(cv=cv, test=bac(y[te], m.predict(Ks[s][np.ix_(te, tr)])), s=s)

def job(name, seed):
    d = np.load(f'{DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    Z = np.nan_to_num(StandardScaler().fit(X[tr]).transform(X))
    F = PCA(n, random_state=0).fit(Z[tr]).transform(Z); F = StandardScaler().fit(F[tr]).transform(F)
    sub = tr[np.random.default_rng(seed).permutation(len(tr))[:200]]   # calibration uses training rows only
    targets = [stat(surrogate(c * F[sub], c * F[sub], G)) for c in C0]
    res, kerns = {}, {}
    res['surr'] = tune_eval([surrogate(c * F, c * F, G) for c in C0], y, tr, te)
    for j, D in enumerate(PERT):
        Gj = np.diag(D) @ G @ np.diag(D)
        cs = calibrate(lambda Z_: surrogate(Z_, Z_, Gj), F[sub], targets)
        res[f'sv{j}'] = tune_eval([surrogate(c * F, c * F, Gj) for c in cs], y, tr, te)
    for i in [lad] + MEM:
        p = perm_of(Gs[i]); f = lambda Z_: fidelity(states(Z_, L, p), states(Z_, L, p))
        cs = calibrate(f, F[sub], targets); Ks = [f(c * F) for c in cs]
        res[f'm{i}'] = tune_eval(Ks, y, tr, te); kerns[i] = Ks[6]
    off = ~np.eye(len(y), dtype=bool)
    res['dist_member'] = float(np.mean([np.abs(kerns[i] - kerns[lad])[off].mean() for i in MEM]))
    Gj = np.diag(PERT[0]) @ G @ np.diag(PERT[0]); cs = calibrate(lambda Z_: surrogate(Z_, Z_, Gj), F[sub], targets[6:7])
    res['dist_surr'] = float(np.abs(surrogate(cs[0] * F, cs[0] * F, Gj) - surrogate(C0[6] * F, C0[6] * F, G))[off].mean())
    return name, seed, res

if __name__ == '__main__':
    out = Parallel(n_jobs=24)(delayed(job)(nm, s) for nm in DEV for s in range(SPLITS))
    json.dump(dict(lad=int(lad), mem=[int(i) for i in MEM], reps=len(reps), out=out), open(P + '/results/pilot2.json', 'w'))
    print('done')
