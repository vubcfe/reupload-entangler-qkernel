# Finite-shot robustness

Fixed 3 Nov 2026, before any finite-shot run. Not externally registered.

## Question

Do the singular-metric loss and the geometric-scale remedy survive kernel estimation from a finite number of
measurement shots?

## Protocol (`src/shots.py`)

- **Data:** all 23 data sets, PCA pipeline, n = 4, L = 3, fresh split seeds 300–304 (5 splits).
- **Arms:**
  - the uniform ladder;
  - the first 10 singular arms of PREREG_CONFIRM.md (seed 11), each with uniform and with geometric
    w = (1, 1/2, 1/4).
- **Kernel estimation:** the kernel is the probability of the all-zero outcome of the compute–uncompute circuit,
  so each off-diagonal entry is estimated as Binomial(S, k)/S, independently per pair. Training and test entries
  are drawn independently; diagonals are set to 1. Shot budgets: S ∈ {100, 1000}, plus the exact kernel.
- **Bandwidth:** each arm's scale is calibrated on the exact kernel as in `src/main.py`. This stands for a
  calibration done once with many shots. Target and C are then chosen by inner CV on the noisy kernel.
- **Noise seeds:** shot noise uses seed 1000·split + arm index.

## Hypotheses

Tested at each shot level S; the unit is the data set, with 90% bootstrap intervals.

- **S1:** mean singular-uniform minus ladder < 0 (the interval excludes 0).
- **S2:** mean singular-geometric minus singular-uniform > 0 (the interval excludes 0).

Also reported: accuracy of every arm against S.
