"""Pilot B: choose the entangler classically by the closed-form metric (no quantum evaluation), then run the quantum
kernel. Arms: ladder, random entanglers, metric-designed entangler, and the designed metric's Gaussian surrogate."""
import sys, pickle, json, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bac
from paths import ROOT as P, DATA; from lin import *
warnings.filterwarnings('ignore')
DEV = ['diabetes', 'ionosphere', 'sonar', 'wdbc', 'vehicle', 'ilpd', 'heart', 'pc1', 'balance', 'phoneme']
n, L = 4, 3; CG = [.1, 1, 10, 100]; C0 = np.geomspace(.05, 1.6, 12)
Gs, cls = pickle.load(open(P + '/results/classes.pkl', 'rb'))
KEYS = list(cls[L]); REP = [cls[L][k][0] for k in KEYS]          # one entangler per metric class
METS = [np.array(k).reshape(n, n) / 4 for k in KEYS]
LAD = next(i for i, M in enumerate(METS) if np.allclose(M, metric(perm_of(ladder(n)), n, L)))

def stat(K):
    off = ~np.eye(len(K), dtype=bool); return np.median(-np.log(np.clip(K[off], 1e-300, 1)))

def align(K, y):
    Y = (y[:, None] == y[None]).astype(float); H = np.eye(len(y)) - 1 / len(y)
    a, b = H @ K @ H, H @ Y @ H; return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())

def scale_for(f, F, t):
    lo, hi = np.log(1e-3), np.log(5.)
    for _ in range(22):
        mid = (lo + hi) / 2
        if stat(f(np.exp(mid) * F)) < t: lo = mid
        else: hi = mid
    return np.exp((lo + hi) / 2)

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
    return dict(cv=cv, test=bac(y[te], m.predict(Ks[s][np.ix_(te, tr)])))

def job(name, seed, pipe):
    d = np.load(f'{DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    Z = np.nan_to_num(StandardScaler().fit(X[tr]).transform(X))
    if pipe == 'pca': F = PCA(n, random_state=0).fit(Z[tr]).transform(Z)
    else: F = Z[:, np.argsort(-mutual_info_classif(Z[tr], y[tr], random_state=seed))[:n]]
    F = StandardScaler().fit(F[tr]).transform(F)
    sub = tr[np.random.default_rng(seed).permutation(len(tr))[:200]]
    # classical design step: rank every metric class by centred alignment of its Gaussian surrogate at matched bandwidth
    t = 1.0; sc = []
    for M in METS:
        c = np.sqrt(t / np.median(np.einsum('ijk,kl,ijl->ij', *(lambda D: (D, M, D))(F[sub][:, None] - F[sub][None]))[~np.eye(len(sub), dtype=bool)]))
        sc.append(align(surrogate(c * F[sub], c * F[sub], M), y[sub]))
    best = int(np.argmax(sc)); rnd = list(np.random.default_rng(seed).choice(len(METS), 3, replace=False))
    targets = [stat(surrogate(c * F[sub], c * F[sub], METS[LAD])) for c in C0]
    res = dict(rank_ladder=int((np.array(sc) > sc[LAD]).sum()), align_best=float(sc[best]), align_ladder=float(sc[LAD]))
    for tag, k in [('ladder', LAD), ('design', best)] + [(f'rand{j}', r) for j, r in enumerate(rnd)]:
        p = perm_of(Gs[REP[k]]); f = lambda Z_: fidelity(states(Z_, L, p), states(Z_, L, p))
        Ks = [f(scale_for(f, F[sub], tt) * F) for tt in targets]; res[tag] = tune_eval(Ks, y, tr, te)
    M = METS[best]; g = lambda Z_: surrogate(Z_, Z_, M)
    res['design_surr'] = tune_eval([g(scale_for(g, F[sub], tt) * F) for tt in targets], y, tr, te)
    res['offdiag_best'] = float(np.abs(M - np.diag(np.diag(M))).sum() / np.trace(M))
    return name, seed, pipe, res

if __name__ == '__main__':
    out = Parallel(n_jobs=24)(delayed(job)(nm, s, p) for nm in DEV for s in range(5) for p in ('pca', 'mi'))
    json.dump(out, open(P + '/results/pilot_design.json', 'w')); print('done')
