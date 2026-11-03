# Remedy test: geometric layer scales

Fixed 3 Nov 2026. Written while the confirmatory run (PREREG_CONFIRM.md) was still running, before any of its
results had been seen. Not externally registered.

## Theory behind it

With equal layer scales, rank(G) < n happens almost always (2140 of the 2152 singular entanglers at L = 3, n = 4)
when two qubits q and r lie on the same cycle of A and the cycle length divides L. Both rows of T then visit the
same states equally often, and G(e_q − e_r) = 0: to second order the kernel only sees x_q + x_r.

Unequal layer scales w_l break this. With w = (1, 1/2, 1/4), every one of the 20160 entanglers has a full-rank
metric, and the smallest eigenvalue of G is at least 0.2·tr(G)/n. The superincreasing property of w (each w_l is
larger than the sum of all later ones) is a candidate sufficient condition, to be proved separately.

## Data and protocol

- **Data:** all 23 data sets, with fresh split seeds 100–109 (10 splits) that no earlier run used. Both
  pipelines (PCA and MI), n = 4, L = 3.
- **Bandwidth:** matched to the same targets as `src/main.py`; target and C are tuned by inner CV.

## Arms (`src/remedy.py`), each with uniform w = (1, 1, 1) and geometric w = (1, 1/2, 1/4)

- The ladder.
- The 20 singular classes of PREREG_CONFIRM.md (seed 11).
- The first 10 of its 20 full-rank classes (seed 12).

## Hypotheses

The unit is the data set; both pipelines stay together in the bootstrap. Units are points of balanced accuracy.

- **R1:** for singular entanglers, geometric minus uniform is positive. The mean over the 20 arms must be above 1,
  with the 90% bootstrap interval excluding 0.
- **R2:** singular entanglers with geometric w are equivalent to the uniform ladder. The 90% interval must lie
  within ±1.
- **R3:** for full-rank entanglers, geometric minus uniform is equivalent. The 90% interval must lie within ±0.5.
- **R4:** the geometric ladder minus the uniform ladder is equivalent. The 90% interval must lie within ±0.5.

## Addendum (3 Nov 2026, written while the remedy run was still running; none of its results seen)

After the second review: on the confirm data, effects among full-rank classes tracked the relative smallest
eigenvalue of G (exploratory; Spearman +0.65). One added prediction is tested on the remedy data:

**R5.** Among the 10 full-rank arms with uniform w, the arm's mean (over all splits, pipelines and data sets) minus
the uniform ladder has a positive Spearman correlation with ρ = λ_min(G)·n/tr(G). The 20 singular arms are
excluded, since their ρ = 0 would make the test trivial. Test: one-sided permutation test of the Spearman
correlation across the 10 arms, 10000 permutations. Also reported: the same correlation over all 31 uniform arms,
and the geometric arms.

Deviation noted: `remedy.py` contains a harmless leftover, `['heart'] * 0`, which adds nothing; the data set list is
the 23 data sets.
