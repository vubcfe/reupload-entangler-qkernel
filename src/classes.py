"""Group all CNOT-only entanglers on n = 4 qubits by their second-order metric (iso-metric classes)."""
import sys, collections, pickle, numpy as np
from paths import ROOT; from lin import *
n = 4; Gs = gl2(n); out = {}
for L in (2, 3, 4, 6):
    cls = collections.defaultdict(list)
    for i, A in enumerate(Gs):
        cls[tuple(np.round(metric(perm_of(A), n, L) * 4).astype(int).ravel())].append(i)
    key = tuple(np.round(metric(perm_of(ladder(n)), n, L) * 4).astype(int).ravel())
    sizes = sorted(map(len, cls.values()), reverse=True)
    perms = {k: len({tuple(perm_of(Gs[i])) for i in v}) for k, v in cls.items()}
    print(f'L={L}: {len(cls)} metric classes over {len(Gs)} entanglers; largest {sizes[:5]}; ladder class size {len(cls[key])}; '
          f'diagonal classes {sum(np.count_nonzero(np.array(k).reshape(n, n) - np.diag(np.diag(np.array(k).reshape(n, n)))) == 0 for k in cls)}')
    out[L] = dict(cls)
pickle.dump((Gs, out), open(ROOT + '/results/classes.pkl', 'wb'))
