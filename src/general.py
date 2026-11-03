"""Generalization run (PREREG_GENERAL.md): n = 4 with L in {2, 4, 6}, and n = 6 with L = 3."""
import sys, os, json, pickle, numpy as np, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import main as M
from lin import *
warnings.filterwarnings('ignore')
P = M.P; Gs, cls = pickle.load(open(P + '/results/classes.pkl', 'rb'))
ALL = sorted(M.HELD + ['diabetes', 'ionosphere', 'sonar', 'wdbc', 'vehicle', 'ilpd', 'heart', 'pc1', 'balance', 'phoneme'])
FEW = {'balance', 'banknote', 'blood', 'iris'}

def arms(n, L):
    if n == 4:
        keys = list(cls[L]); rk = [np.linalg.matrix_rank(np.array(k).reshape(4, 4)) for k in keys]
        lkey = tuple(np.round(metric(perm_of(ladder(4)), 4, L) * 4).astype(int).ravel())
        S = [cls[L][keys[j]][0] for j in np.random.default_rng(21).choice([j for j in range(len(keys)) if rk[j] < 4], 10, replace=False)]
        F = [cls[L][keys[j]][0] for j in np.random.default_rng(22).choice([j for j in range(len(keys)) if rk[j] == 4 and keys[j] != lkey], 10, replace=False)]
        return [Gs[i] for i in S], [Gs[i] for i in F]
    out = []
    for seed, want in ((23, True), (24, False)):
        rng, got = np.random.default_rng(seed), []
        while len(got) < 10:
            A = rng.integers(0, 2, (n, n))
            if round(abs(np.linalg.det(A))) % 2 == 0: continue
            if (np.linalg.matrix_rank(metric(perm_of(A), n, L)) < n) == want: got.append(A)
        out.append(got)
    return out

SETTINGS = [(4, 2), (4, 4), (4, 6), (6, 3)]
ARMS = {s: arms(*s) for s in SETTINGS}

def job(name, seed):
    fn = f'{P}/results/general/{name}_{seed}.json'
    if os.path.exists(fn): return
    d = np.load(f'{M.DATA}/{name}.npz'); X, y = d['X'], d['y']
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    Z = np.nan_to_num(StandardScaler().fit(X[tr]).transform(X)); res = {}
    for n, L in SETTINGS:
        if n > X.shape[1]: continue
        F = PCA(n, random_state=0).fit(Z[tr]).transform(Z); F = StandardScaler().fit(F[tr]).transform(F)
        sub = tr[np.random.default_rng(seed).permutation(len(tr))[:M.NSUB]]
        G0 = metric(perm_of(ladder(n)), n, L); targets = [M.stat(surrogate(c * F[sub], c * F[sub], G0)) for c in M.C0]
        S, Fr = ARMS[(n, L)]; geo = 2.0 ** -np.arange(L)
        todo = [('lad_u', ladder(n), None)] + [(f's{j}_u', A, None) for j, A in enumerate(S)] + \
               [(f's{j}_g', A, geo) for j, A in enumerate(S)] + [(f'f{j}_u', A, None) for j, A in enumerate(Fr)]
        for tag, A, w in todo:
            p = perm_of(A); f = lambda Z_: fidelity(states(Z_, L, p, w), states(Z_, L, p, w))
            cs = M.calibrate(f, F[sub], targets); r = M.tune_eval([f(c * F) for c in cs], y, tr, te)
            Gm = metric(p, n, L, w); r['rho'] = float(np.linalg.eigvalsh(Gm)[0] * n / np.trace(Gm))
            res[f'n{n}L{L}_{tag}'] = r
    json.dump(res, open(fn, 'w'))

if __name__ == '__main__':
    print({s: (len(a[0]), len(a[1])) for s, a in ARMS.items()})
    Parallel(n_jobs=24)(delayed(job)(nm, s) for s in range(200, 208) for nm in ALL)
    print('done')
