"""Check the general metric against simulation, including entanglers with non-diagonal G."""
import sys, numpy as np
from lin import *
n = 4
# the ladder CX(n-2,n-1)...CX(0,1), applied gate by gate on basis indices (qubit 0 = most significant bit)
idx = np.arange(2**n)
for q in range(n - 1): idx = np.where((idx >> (n - 1 - q)) & 1, idx ^ (1 << (n - 2 - q)), idx)
assert (perm_of(ladder(n)) == idx).all()
rng = np.random.default_rng(0); Gs = gl2(n); print('GL(4,2) size', len(Gs))
worst = 0; nondiag = 0
for trial in range(200):
    A = Gs[rng.integers(len(Gs))]; L = int(rng.integers(1, 7)); w = rng.uniform(.3, 2, L); p = perm_of(A)
    G = metric(p, n, L, w); nondiag += np.abs(G - np.diag(np.diag(G))).max() > 0
    X = rng.normal(size=(30, n)); c = 1e-2
    K = fidelity(states(c * X, L, p, w), states(c * X, L, p, w))
    D = X[:, None] - X[None]; Q = np.einsum('ijk,kl,ijl->ij', D, G, D) * c * c
    off = ~np.eye(30, dtype=bool); worst = max(worst, np.max(np.abs((1 - K[off]) - Q[off]) / Q[off]))
print('non-diagonal metrics:', nondiag, '/ 200;  worst relative error of 1-k vs quadratic form at c=0.01:', worst)
