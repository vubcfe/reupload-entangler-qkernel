"""Finite-shot robustness run (PREREG_SHOTS.md)."""
import sys, os, json, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import main as M, confirm as CF
from lin import *
warnings.filterwarnings('ignore')
P = M.P; n, L = 4, 3
ALL = sorted(M.HELD + CF.DEV)
ARMS = [('lad_u', M.LAD, None)] + [(f's{j}_u', M.cls[L][M.KEYS[j]][0], None) for j in CF.S_ARMS[:10]] + \
       [(f's{j}_g', M.cls[L][M.KEYS[j]][0], np.array([1, .5, .25])) for j in CF.S_ARMS[:10]]
SHOTS = [0, 100, 1000]

def noisy(K, S, rng):
    if S == 0: return K
    N = rng.binomial(S, np.clip(K, 0, 1)) / S; N = np.triu(N, 1); N = N + N.T; np.fill_diagonal(N, 1); return N

def job(name, seed):
    fn = f'{P}/results/shots/{name}_{seed}.json'
    if os.path.exists(fn): return
    d = np.load(f'{M.DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    Z = np.nan_to_num(StandardScaler().fit(X[tr]).transform(X))
    F = PCA(n, random_state=0).fit(Z[tr]).transform(Z); F = StandardScaler().fit(F[tr]).transform(F)
    sub = tr[np.random.default_rng(seed).permutation(len(tr))[:M.NSUB]]
    targets = [M.stat(surrogate(c * F[sub], c * F[sub], M.G)) for c in M.C0]; res = {}
    for a, (tag, i, w) in enumerate(ARMS):
        p = perm_of(M.Gs[i]); f = lambda Z_: fidelity(states(Z_, L, p, w), states(Z_, L, p, w))
        cs = M.calibrate(f, F[sub], targets); Ks = [f(c * F) for c in cs]
        for S in SHOTS:
            rng = np.random.default_rng(1000 * seed + a)
            res[f'{tag}_S{S}'] = M.tune_eval([noisy(K, S, rng) for K in Ks], y, tr, te)
    json.dump(res, open(fn, 'w'))

if __name__ == '__main__':
    Parallel(n_jobs=24)(delayed(job)(nm, s) for s in range(300, 305) for nm in ALL)
    print('done')
