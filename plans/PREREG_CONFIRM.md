# Confirmatory test of the singular-metric rule

Fixed 3 Nov 2026, after the held-out main run. Nothing has been run on the development data sets for this
hypothesis. Not externally registered.

## Background

In the main run (held-out data), the 4 of 30 sampled metric classes with rank(G) < 4 lost 1.6–5.2 points against
the CNOT ladder. No full-rank class beat the ladder by more than 0.5 points. This was found after looking at the
data (exploratory). Here it is tested on independent data.

## Data and protocol

- **Data:** the 10 development sets (diabetes, ionosphere, sonar, wdbc, vehicle, ilpd, heart, pc1, balance,
  phoneme). Pilots on these sets compared only members of the ladder's class (pilot 2) and one alignment-chosen
  class per split (pilot B). Neither singled out singular metrics.
- **Splits and pipelines:** 15 splits (seeds 0–14), PCA and MI pipelines, n = 4, L = 3.
- **Bandwidth matching and tuning:** exactly as in `src/main.py`.

## Arms (`src/confirm.py`)

- The ladder.
- 20 singular classes drawn at random (seed 11) from the 87 with rank(G) < 4, one entangler each.
- 20 full-rank classes drawn at random (seed 12) from the 550 with rank(G) = 4, excluding the ladder's class.

## Hypotheses

The unit is the data set; both pipelines of a data set are kept together in the bootstrap. Units are points of
balanced accuracy.

- **C1:** singular classes score below the ladder. Statistic: the mean over singular arms of arm minus ladder.
  Supported if the 90% bootstrap interval lies entirely below 0 **and** the mean is below −1.
- **C2:** full-rank classes are equivalent to the ladder. Statistic: the mean over full-rank arms of arm minus
  ladder. Supported if the 90% interval lies within ±0.5.
- **C3:** singular minus full-rank, using the per-data-set means of the two groups. Predicted below 0. Test:
  one-sided Wilcoxon over data sets.
- **C4 (mechanism):** the loss of a singular class is larger when the direction its metric ignores carries label
  information.
  - Predictor, classical and computed on training rows only: d = BA(RBF on F) − BA(RBF on F projected onto the
    range of G). Both terms come from 3-fold CV with γ and C tuned on the grids γ ∈ {1e-3, 1e-2, 0.03, 0.1, 0.3, 1, 3} and C ∈ {0.1, 1, 10, 100}.
  - Prediction: across singular arms × cells, the Spearman correlation between d and the arm's loss against the
    ladder is negative (larger d, more negative arm − ladder).
  - Test: per data set, the sign of the within-data-set Spearman correlation; one-sided sign test.

## Also reported

- How the share of singular entanglers depends on L and n.
- The theoretical condition for rank(G) < n.
