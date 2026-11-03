"""Re-uploading kernels whose entangler is any linear reversible (CNOT-only) circuit, given as A in GL(n, 2)."""
import itertools, numpy as np

def bits(n):
    """Row i holds the bits of basis index i; qubit 0 is the most significant bit."""
    i = np.arange(2**n)
    return (i[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1

def perm_of(A):
    """Basis permutation of the circuit acting as b -> A b over GF(2)."""
    n = len(A); B = bits(n)
    return ((B @ A.T) % 2) @ (1 << (n - 1 - np.arange(n)))

def ladder(n):
    return np.tril(np.ones((n, n), int))

def gl2(n):
    """All invertible n x n binary matrices (20160 for n = 4)."""
    out = []
    for rows in itertools.product(range(1, 2**n), repeat=n):
        M = ((np.array(rows)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)
        if round(abs(np.linalg.det(M))) % 2 == 1 and _rank2(M) == n: out.append(M)
    return out

def _rank2(M):
    M = M.copy() % 2; r = 0
    for c in range(M.shape[1]):
        p = [i for i in range(r, len(M)) if M[i, c]]
        if not p: continue
        M[[r, p[0]]] = M[[p[0], r]]
        for i in range(len(M)):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
    return r

def _rot(st, q, n, a, b, c, d):
    m = st.shape[0]
    s = st.reshape(m, 2**q, 2, 2**(n - q - 1))
    s0, s1 = s[:, :, 0, :].copy(), s[:, :, 1, :].copy()
    s[:, :, 0, :] = a[:, None, None] * s0 + b[:, None, None] * s1
    s[:, :, 1, :] = c[:, None, None] * s0 + d[:, None, None] * s1
    return s.reshape(m, 2**n)

def states(x, L, perm, w=None):
    """L layers of RZ then RX on every qubit (layer scale w[l]), each followed by the entangler."""
    m, n = x.shape
    w = np.ones(L) if w is None else w
    st = np.zeros((m, 2**n), complex); st[:, 0] = 1
    inv = np.argsort(perm)
    for l in range(L):
        for q in range(n):
            t = w[l] * x[:, q]
            st = _rot(st, q, n, np.exp(-.5j * t), 0 * t, 0 * t, np.exp(.5j * t))
            co, si = np.cos(t / 2), -1j * np.sin(t / 2)
            st = _rot(st, q, n, co, si, si, co)
        st = st[:, inv]
    return st

def fidelity(a, b):
    return np.abs(a.conj() @ b.T) ** 2

def metric(perm, n, L, w=None):
    """G = T T^T / 4, T[q, s] = sum of w_l over layers l whose e_q term sits on basis state s at the end."""
    w = np.ones(L) if w is None else w
    T = np.zeros((n, 2**n))
    for q in range(n):
        for l in range(L):
            s = 1 << (n - 1 - q)
            for _ in range(L - l): s = perm[s]
            T[q, s] += w[l]
    return T @ T.T / 4

def surrogate(A, B, G):
    D = A[:, None, :] - B[None, :, :]
    return np.exp(-np.einsum('ijk,kl,ijl->ij', D, G, D))
