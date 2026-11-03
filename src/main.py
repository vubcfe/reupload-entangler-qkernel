"""Main run (PREREGISTRATION.md, revision 2): does the CNOT entangler matter once bandwidth is matched?
One job = data set x split x pipeline; writes results/main/{name}_{seed}_{pipe}.json."""
import sys, os, pickle, json, itertools, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bac
from paths import ROOT as P, DATA; from lin import *
warnings.filterwarnings('ignore')
HELD = ['banknote', 'blood', 'climate', 'iris', 'kc1', 'magic', 'optdigits', 'qsar', 'segment', 'spambase', 'steel', 'wilt', 'wine']
n, L, SPLITS = 4, 3, 15; CG = [.1, 1, 10, 100]; C0 = np.geomspace(.05, 1.6, 12); NSUB = 200
Gs, cls = pickle.load(open(P + '/results/classes.pkl', 'rb'))
G = metric(perm_of(ladder(n)), n, L); LKEY = tuple(np.round(G * 4).astype(int).ravel())
stab = [np.eye(n, dtype=int)[list(p)] for p in itertools.permutations(range(n))
        if np.allclose(np.eye(n)[list(p)] @ G @ np.eye(n)[list(p)].T, G)]
look = {Gs[i].tobytes(): i for i in cls[L][LKEY]}
orb = lambda i: min(look[(Q @ Gs[i] @ Q.T).tobytes()] for Q in stab)
LAD = orb(look[ladder(n).tobytes()])
T1 = sorted({orb(i) for i in cls[L][LKEY]} - {LAD})                        # 30 orbit representatives
KEYS = list(cls[L]); METS = [np.array(k).reshape(n, n) / 4 for k in KEYS]
other = [j for j, k in enumerate(KEYS) if k != LKEY]
T2C = list(np.random.default_rng(7).choice(other, 30, replace=False)); T2 = [cls[L][KEYS[j]][0] for j in T2C]
PERT = [np.exp(np.random.default_rng(100 + j).normal(0, .18, n)) for j in range(30)]

def stat(K):
    off = ~np.eye(len(K), dtype=bool); return np.median(-np.log(np.clip(K[off], 1e-300, 1)))

def align(K, y):
    Y = (y[:, None] == y[None]).astype(float); H = np.eye(len(y)) - 1 / len(y)
    a, b = H @ K @ H, H @ Y @ H; return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())

def calibrate(f, F, targets):
    out = []
    for t in targets:
        lo, hi = np.log(1e-3), np.log(5.)
        for _ in range(22):
            mid = (lo + hi) / 2
            if stat(f(np.exp(mid) * F)) < t: lo = mid
            else: hi = mid
        out.append(float(np.exp((lo + hi) / 2)))
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
    return dict(cv=float(cv), test=float(bac(y[te], m.predict(Ks[s][np.ix_(te, tr)]))), s=int(s))

def job(name, seed, pipe):
    fn = f'{P}/results/main/{name}_{seed}_{pipe}.json'
    if os.path.exists(fn): return
    d = np.load(f'{DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    Z = np.nan_to_num(StandardScaler().fit(X[tr]).transform(X))
    if pipe == 'pca': F = PCA(n, random_state=0).fit(Z[tr]).transform(Z)
    else: F = Z[:, np.argsort(-mutual_info_classif(Z[tr], y[tr], random_state=seed))[:n]]
    F = StandardScaler().fit(F[tr]).transform(F)
    sub = tr[np.random.default_rng(seed).permutation(len(tr))[:NSUB]]; m = len(sub); off = ~np.eye(len(y), dtype=bool)
    targets = [stat(surrogate(c * F[sub], c * F[sub], G)) for c in C0]
    res = {}; Ks_surr = [surrogate(c * F, c * F, G) for c in C0]; res['surr'] = tune_eval(Ks_surr, y, tr, te)
    for j, D in enumerate(PERT):
        Gj = np.diag(D) @ G @ np.diag(D); cs = calibrate(lambda Z_: surrogate(Z_, Z_, Gj), F[sub], targets)
        res[f'sv{j}'] = tune_eval([surrogate(c * F, c * F, Gj) for c in cs], y, tr, te)
    # classical design step over all metric classes (training rows only)
    D2 = F[sub][:, None] - F[sub][None]; sc = []
    for M in METS:
        q = np.einsum('ijk,kl,ijl->ij', D2, M, D2); c = np.sqrt(1. / np.median(q[~np.eye(m, dtype=bool)]))
        sc.append(align(surrogate(c * F[sub], c * F[sub], M), y[sub]))
    DES = cls[L][KEYS[int(np.argmax(sc))]][0]; res['design_class'] = int(np.argmax(sc))
    arms = [('lad', LAD)] + [(f'w{i}', i) for i in T1] + [(f'b{i}', i) for i in T2] + [('des', DES)]
    for tag, i in arms:
        p = perm_of(Gs[i]); f = lambda Z_: fidelity(states(Z_, L, p), states(Z_, L, p))
        cs = calibrate(f, F[sub], targets); Ks = [f(c * F) for c in cs]
        res[tag] = tune_eval(Ks, y, tr, te); res[tag]['cs'] = cs
        if tag.startswith('w'): res[tag]['dist'] = float(np.abs(Ks[6] - Ks_surr[6])[off].mean())
        if tag == 'lad' or tag.startswith('b'): res[tag + '_fix'] = tune_eval([f(1.0 * F)], y, tr, te)
    json.dump(res, open(fn, 'w'))

if __name__ == '__main__':
    Parallel(n_jobs=24)(delayed(job)(nm, s, p) for s in range(SPLITS) for nm in HELD for p in ('pca', 'mi'))
    print('done')
