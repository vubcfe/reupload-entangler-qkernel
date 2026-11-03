"""Confirmatory run for the singular-metric rule (PREREG_CONFIRM.md) on the development data sets."""
import sys, os, json, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bac
import main as M
from lin import *
warnings.filterwarnings('ignore')
P = M.P; n, L = 4, 3
DEV = ['diabetes', 'ionosphere', 'sonar', 'wdbc', 'vehicle', 'ilpd', 'heart', 'pc1', 'balance', 'phoneme']
rank = [np.linalg.matrix_rank(Mt) for Mt in M.METS]
SING = [j for j in range(len(M.KEYS)) if rank[j] < n]; FULL = [j for j in range(len(M.KEYS)) if rank[j] == n and M.KEYS[j] != M.LKEY]
S_ARMS = list(np.random.default_rng(11).choice(SING, 20, replace=False)); F_ARMS = list(np.random.default_rng(12).choice(FULL, 20, replace=False))
GAMMA = [1e-3, 1e-2, .03, .1, .3, 1, 3]

def rbf_cv(F, y, tr):
    """Best 3-fold CV balanced accuracy of an RBF SVM over (gamma, C), training rows only."""
    D = ((F[tr][:, None] - F[tr][None]) ** 2).sum(-1); best = 0
    for g in GAMMA:
        K = np.exp(-g * D)
        for C in M.CG:
            sc = []
            for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, y[tr]):
                m = SVC(kernel='precomputed', C=C, class_weight='balanced').fit(K[np.ix_(a, a)], y[tr[a]])
                sc.append(bac(y[tr[b]], m.predict(K[np.ix_(b, a)])))
            best = max(best, np.mean(sc))
    return best

def job(name, seed, pipe):
    fn = f'{P}/results/confirm/{name}_{seed}_{pipe}.json'
    if os.path.exists(fn): return
    d = np.load(f'{M.DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    Z = np.nan_to_num(StandardScaler().fit(X[tr]).transform(X))
    if pipe == 'pca': F = PCA(n, random_state=0).fit(Z[tr]).transform(Z)
    else: F = Z[:, np.argsort(-mutual_info_classif(Z[tr], y[tr], random_state=seed))[:n]]
    F = StandardScaler().fit(F[tr]).transform(F)
    sub = tr[np.random.default_rng(seed).permutation(len(tr))[:M.NSUB]]
    targets = [M.stat(surrogate(c * F[sub], c * F[sub], M.G)) for c in M.C0]
    res = {'rbf_full': rbf_cv(F, y, tr)}
    for tag, i in [('lad', M.LAD)] + [(f's{j}', M.cls[L][M.KEYS[j]][0]) for j in S_ARMS] + [(f'f{j}', M.cls[L][M.KEYS[j]][0]) for j in F_ARMS]:
        p = perm_of(M.Gs[i]); f = lambda Z_: fidelity(states(Z_, L, p), states(Z_, L, p))
        cs = M.calibrate(f, F[sub], targets); res[tag] = M.tune_eval([f(c * F) for c in cs], y, tr, te)
        if tag[0] == 's':
            w, V = np.linalg.eigh(M.METS[int(tag[1:])]); R = V[:, w > 1e-9 * w.max()]
            res[tag]['rbf_proj'] = rbf_cv(F @ R @ R.T, y, tr)
    json.dump(res, open(fn, 'w'))

if __name__ == '__main__':
    print(len(SING), 'singular,', len(FULL), 'full-rank classes')
    Parallel(n_jobs=24)(delayed(job)(nm, s, p) for s in range(15) for nm in DEV for p in ('pca', 'mi'))
    print('done')
