"""Remedy run (PREREG_REMEDY.md): uniform vs geometric layer scales for singular, full-rank and ladder entanglers."""
import sys, os, json, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import main as M, confirm as CF
from lin import *
warnings.filterwarnings('ignore')
P = M.P; n, L = 4, 3
ALL = sorted(M.HELD + CF.DEV)
WS = {'u': np.ones(3), 'g': np.array([1, .5, .25])}
ARMS = [('lad', M.LAD)] + [(f's{j}', M.cls[L][M.KEYS[j]][0]) for j in CF.S_ARMS] + [(f'f{j}', M.cls[L][M.KEYS[j]][0]) for j in CF.F_ARMS[:10]]

def job(name, seed, pipe):
    fn = f'{P}/results/remedy/{name}_{seed}_{pipe}.json'
    if os.path.exists(fn): return
    d = np.load(f'{M.DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    Z = np.nan_to_num(StandardScaler().fit(X[tr]).transform(X))
    if pipe == 'pca': F = PCA(n, random_state=0).fit(Z[tr]).transform(Z)
    else: F = Z[:, np.argsort(-mutual_info_classif(Z[tr], y[tr], random_state=seed))[:n]]
    F = StandardScaler().fit(F[tr]).transform(F)
    sub = tr[np.random.default_rng(seed).permutation(len(tr))[:M.NSUB]]
    targets = [M.stat(surrogate(c * F[sub], c * F[sub], M.G)) for c in M.C0]
    res = {}
    for tag, i in ARMS:
        p = perm_of(M.Gs[i])
        for wn, w in WS.items():
            f = lambda Z_: fidelity(states(Z_, L, p, w), states(Z_, L, p, w))
            cs = M.calibrate(f, F[sub], targets); res[f'{tag}_{wn}'] = M.tune_eval([f(c * F) for c in cs], y, tr, te)
    json.dump(res, open(fn, 'w'))

if __name__ == '__main__':
    print(len(ALL), 'data sets')
    Parallel(n_jobs=24)(delayed(job)(nm, s, p) for s in range(100, 110) for nm in ALL for p in ('pca', 'mi'))
    print('done')
