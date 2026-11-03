# Theory notes (3 Nov 2026)

## Setting

- n qubits; A ∈ GL(n,2) is a CNOT-only entangler acting on basis states as b ↦ Ab.
- Each of the L layers applies RZ(w_l c x_q) and then RX(w_l c x_q) to every qubit, followed by A.
- The kernel is k(x,x') = |⟨ψ(x)|ψ(x')⟩|².

## Lemma 1 (second-order metric)

k(x,x') = 1 − c² Δᵀ G Δ + O(c⁴), with G = T Tᵀ/4, where

T[q,s] = Σ_l w_l · [A^{L−l+1} e_q = s].

**Proof idea.**
- Push every RX generator forward through the remaining entanglers: X_q becomes X^{A^{L−l+1}e_q}.
- On |0⟩, Cov(X^s, X^{s'}) = [s = s'].
- RZ terms stay Z-type and do not enter at second order.
- Odd orders vanish because ψ(−x) = ψ(x)*.
- This is the Fubini–Study (quantum Fisher) metric of Pauli generators; it is standard for Clifford-conjugated
  Paulis, so we state it as a lemma.

**Checks.**
- `src/check_metric.py`: 200 random (A, L, w); the relative error scales as c².
- The CNOT ladder with w = 1 gives a diagonal metric.

## Enumeration (n = 4)

Number of distinct metrics among the 20160 entanglers:

| L | metric classes | singular classes | singular entanglers |
|---|---|---|---|
| 2 | 123 | 36 | 2.9% |
| 3 | 638 | 87 | 10.7% |
| 4 | 1230 | 169 | 14.9% |
| 6 | 1516 | 285 | 27.6% |

With w = 1, the ladder's class at L = 3 has 54 members in 31 orbits; the stabilizer of G is the swap of qubits 0
and 1.

## Theorem 2 (exact characterization of singular metrics, uniform w)

A permutes F_2^n \ {0} into cycles. Write T_q[s] = #{k in 1..L : A^k e_q = s}. Then:

- If e_q lies on a cycle C of length p, T_q is supported on C. It is the window indicator of the L positions after
  e_q, wrapped around C.
- Rows of qubits on different cycles have disjoint supports, so rank(T) is the sum over cycles.
- On one cycle, the rows are cyclic shifts S^{i_q} u of one vector u. Its DFT is û(k) = Σ_{m=1..L} ω^{mk}, with
  ω = e^{2πi/p}. û(k) vanishes exactly when k ≠ 0 and kL ≡ 0 (mod p), that is k ∈ {j·p/g : j = 1..g−1}, where
  g = gcd(p, L).
- Σ_q a_q S^{i_q} u = 0 holds iff the coefficient function a (on positions of C, zero off unit vectors) has a DFT
  supported on that zero set. Equivalently, a is g-periodic with zero sum over a period.

**Statement.** G is singular iff some cycle of A has at least two residue classes mod g = gcd(p, L) whose
positions are all unit vectors e_q.

**Corollaries.**
- (Previous Proposition 2.) With p | L, g = p, so the classes are single positions: two unit vectors on one cycle
  of length dividing L make G singular.
- The share of singular entanglers grows with L, because gcd(p, L) tends to grow with L.

**Check.** All 20160 entanglers at n = 4, for L = 2..6: 0 mismatches between the criterion and rank(G).

## Theorem 3 (dominant layer guarantees a positive-definite metric; tight constant)

**Statement.** Suppose some layer ℓ* satisfies w_ℓ* > Σ_{l≠ℓ*} w_l, and let δ = w_ℓ* − Σ_{l≠ℓ*} w_l. Then for every
A ∈ GL(n,2),

λ_min(G) ≥ δ²/4,

independent of n.

**Proof.**
1. Let S be the n × n submatrix of T on the distinct columns s_q = A^{L−ℓ*+1} e_q.
2. Row q: the diagonal entry is ≥ w_ℓ*. Revisits only add to the diagonal and remove off-diagonal mass, so the
   off-diagonal mass is ≤ Σ_{l≠ℓ*} w_l.
3. Column r: for each layer l, the map q ↦ A^{L−l+1} e_q is injective, so at most one off-diagonal entry per layer
   lands in column r. The column's off-diagonal mass is also ≤ Σ_{l≠ℓ*} w_l.
4. By Varah's bound in both norms, ‖S⁻¹‖_∞, ‖S⁻¹‖_1 ≤ 1/δ, so ‖S⁻¹‖_2 ≤ 1/δ.
5. Since TTᵀ ⪰ SSᵀ, λ_min(G) ≥ δ²/4.

**Check.** 4000 random (A, L ≤ 6, w with a dominant layer): the minimum of λ_min/(δ²/4) is 1.0000, so the bound
is tight.

**Example.** w = (1, 1/2, 1/4) has δ = 1/4, so λ_min(G) ≥ 1/64 for every entangler. Uniform scales have no
dominant layer when L ≥ 2.

## Open

- The fourth-order term.
- A characterization of all realizable metrics.
- n > 4 enumeration by sampling: at n = 6, classes of about 800 members exist.
