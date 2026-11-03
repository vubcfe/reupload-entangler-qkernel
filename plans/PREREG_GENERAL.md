# Generalization to other depths and to six qubits

Fixed 3 Nov 2026, before any run with these settings. Not externally registered.

## Settings

- n = 4 with L ∈ {2, 4, 6}.
- n = 6 with L = 3.

## Data

- All 23 data sets, PCA pipeline only, with fresh split seeds 200–207 (8 splits).
- For n = 6, only the data sets with at least 6 features (balance, banknote, blood and iris are dropped).
- Bandwidth matching and tuning are the same as in `src/main.py`. Targets come from the ladder surrogate of each
  setting.

## Arms (`src/general.py`), per setting

| Arm | Count | Layer scales |
|---|---|---|
| CNOT ladder | 1 | uniform |
| Singular entanglers | 10 | uniform, and again with geometric w_l = 2^{−(l−1)} |
| Full-rank entanglers | 10 | uniform |

- **n = 4:** the singular and full-rank entanglers are 10 metric classes each, drawn from that L's classes (seeds
  21 and 22).
- **n = 6:** they are 10 random invertible A each with rank(G) < 6 and rank(G) = 6, drawn by rejection sampling
  (seeds 23 and 24).

## Hypotheses

Tested separately in each of the 4 settings; the unit is the data set, with 90% bootstrap intervals.

- **G1:** singular minus ladder, mean over singular arms, is below 0 (the interval excludes 0).
- **G2:** singular with geometric w minus singular with uniform w is above 0 (the interval excludes 0).
- **G3 (theory, no data):** Theorem 2 predicts the share of singular entanglers. At n = 4 it is 2.9% (L = 2),
  10.7% (L = 3), 14.9% (L = 4) and 27.6% (L = 6). At n = 6 it is estimated by sampling.

Also reported:
- full-rank minus ladder;
- across arms within each setting, Spearman(ρ, arm − ladder), where ρ = λ_min(G)·n/tr(G).
Deviation (3 Nov): phoneme has 5 features, so it is also dropped for n = 6 (prereg listed only the four 4-feature sets). First general run crashed on it after 3 jobs; rerun skips any data set with fewer than n features.
