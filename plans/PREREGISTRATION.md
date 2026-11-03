# Analysis plan for the main run

Fixed 3 Nov 2026, after two pilots on the development data sets and before any run on the held-out data sets.
Not externally registered. Changes after this date will be listed as deviations in the paper.

## What the pilots showed (development data only)

Development sets: diabetes, ionosphere, sonar, wdbc, vehicle, ilpd, heart, pc1, balance, phoneme.

- **Pilot 1** used a coarse shared c grid and no bandwidth matching. The member chosen by inner CV scored +0.46
  points above the Gaussian surrogate. An independent review showed this is confounded: at the selected c, the
  effective bandwidth differs by 10–20% between members of one iso-metric class.
- **Pilot 2** (`src/pilot2.py`) matched every kernel's bandwidth to the same label-free targets. It used one member
  per symmetry orbit and an equal-budget family of 12 perturbed surrogates, with 10 splits. Results:
  - Member effect: pooled within-split permutation p = 0.84. Split-half Spearman of member rankings: mean 0.12.
  - Selected member minus selected surrogate: −0.11 points (6/10 data sets, Wilcoxon p = 0.85).
  - Ladder minus surrogate: −0.46 points.

The pilots therefore point to **no beyond-metric effect once bandwidth is matched**. The main run is designed to
confirm or refute this on unseen data with enough power to bound the effect.

## Revision 2 (3 Nov 2026, still before any held-out run)

Pilot B (`src/pilot_design.py`, development data only) chose the entangler classically by the surrogate's
alignment over all 638 metric classes. It did not beat the ladder (PCA −0.70, MI −0.37 points), and random
entanglers scored about the same as the ladder. The paper is therefore reframed as a test of whether the entangler
matters at all once bandwidth is matched. The design below replaces the earlier n = 4 design; the n = 6 run is
deferred to a separate exploratory study.

## Data

Held-out sets: banknote, blood, climate, iris, kc1, magic, optdigits, qsar, segment, spambase, steel, wilt, wine
(13). Preprocessing as in `src/get_data.py`.

- 15 stratified 80/20 outer splits, seeds 0–14.
- Two pipelines: PCA to 4 components, and the top 4 features by mutual information. Both are fitted on the outer
  training part and restandardized.
- n = 4, L = 3 throughout.

## Arms (`src/main.py`)

- **T1, within class:**
  - the ladder;
  - one representative of each of the other 30 symmetry orbits of the ladder's iso-metric class;
  - the Gaussian surrogate;
  - 30 perturbed surrogates (σ = 0.18, seeds 100–129).
- **T2, between classes:**
  - 30 metric classes drawn at random (seed 7) from the 638, excluding the ladder's class, one entangler each;
  - the design arm: the class with the highest surrogate alignment on 200 training rows (as in pilot B).
- **Bandwidth matching.** Every arm hits the same 12 label-free targets, the median off-diagonal −log k of the
  surrogate at c ∈ geomspace(0.05, 1.6, 12). C ∈ {0.1, 1, 10, 100}. The target and C are chosen by 3-fold inner CV
  on balanced accuracy.
- **T3, search as in the literature.** For the ladder and the 30 T2 entanglers, also evaluate at the fixed scale
  c = 1 with only C tuned. "Search" means choosing the entangler with the best inner-CV score. This is done twice:
  at the fixed scale, and with matched bandwidth.

## Hypotheses

The unit is the data set × pipeline cell, with test balanced accuracy in points. Bootstrap resamples data sets,
keeping both pipelines of a data set together.

- **H1, no within-class effect.**
  - Test: the within-split member-label permutation test on the T1 members, statistic summed over cells, 5000
    permutations.
  - Supported if p > 0.05 and the 90% bootstrap upper bound on the member SD is below 0.5 points.
- **H2, no between-class effect.** The same test and bound on the 30 T2 entanglers, with the ladder added.
- **H3, classical design does not help.**
  - Comparison: the design arm minus the ladder.
  - Equivalence: the 90% bootstrap interval lies within ±1 point. Also reported: Wilcoxon over data sets.
- **H4, search gains are a bandwidth effect.** Let Δ_fix be the search winner minus the ladder at c = 1, and
  Δ_bw the same with matched bandwidth.
  - Predicted: Δ_bw is within ±1 (equivalence), and Δ_fix − Δ_bw > 0 (one-sided Wilcoxon over data sets).
  - Also reported: the best fixed-scale search result minus the best bandwidth-matched result.
- **Secondary, from the first version.** Within-cell Spearman correlation of member kernel distance and accuracy,
  predicted negative, tested by a sign test.

Reported, not tested: the ladder minus the surrogate; the T1 surrogate-family selection minus the T1 member
selection; the selected targets per arm.

## Not in this run

- The closed-form fourth-order term and a fourth-order classical surrogate. These are theory work and will be
  added as a separate arm only if derived before the main run starts. Otherwise they appear in the paper as
  analysis, not as a tested arm.
- Hardware and shot noise.
