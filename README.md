# Which entangler matters in a data re-uploading quantum kernel?

Code, analysis plans and per-split results for the paper *Which entangler matters in a data re-uploading quantum
kernel? Singular metrics, a provably full-rank layer schedule, and matched-bandwidth tests over all CNOT feature maps*
(Cao Vu Bui, Van-Nam Nguyen, Hoang-Anh-Vu Nguyen, Thien-Nhat-Quang Le).

The circuit has L layers of single-qubit R_Z then R_X encodings, with layer scales w_l, each followed by a CNOT-only
entangler A ∈ GL(n, 2). At small bandwidth the fidelity kernel matches a Gaussian kernel with metric
G = T Tᵀ / 4, where T counts the orbit visits of A. The repository contains:

- **Theory checks.** The metric formula, the exact criterion for singular G, and the dominant-layer bound
  λ_min(G) ≥ δ²/4 (see `THEORY.md`).
- **Experiments.** Five studies on 23 OpenML data sets, with bandwidth matched across circuits. Each study's
  analysis plan was written before it was run (`plans/`).

## Setup

```
pip install -r requirements.txt
python src/get_data.py        # downloads the 23 OpenML data sets into data/ (about 1 minute)
python src/classes.py         # enumerates all 20160 four-qubit entanglers into results/classes.pkl
```

## Contents

| Path | Purpose |
|---|---|
| `src/lin.py` | State-vector simulation of the kernel for any A ∈ GL(n, 2), the metric G, and the Gaussian surrogate |
| `src/check_metric.py` | Checks G against simulation (relative error ∝ c²) |
| `src/classes.py` | Metric classes and singular shares at n = 4 (Table 1 of the paper) |
| `src/pilot*.py`, `src/analyze_pilot2.py` | Exploratory pilots on the development data sets |
| `src/main.py`, `src/analyze_main.py` | Main study, H1–H4 (`plans/PREREGISTRATION.md`) |
| `src/confirm.py` | Confirmation of the singular-metric rule, C1–C4 (`plans/PREREG_CONFIRM.md`) |
| `src/remedy.py` | Geometric layer scales, R1–R5 (`plans/PREREG_REMEDY.md`) |
| `src/analyze_prereg2.py` | Analysis for the confirmation and remedy studies |
| `src/general.py` | L ∈ {2, 4, 6} at n = 4, and n = 6 at L = 3, G1–G2 (`plans/PREREG_GENERAL.md`) |
| `src/shots.py` | Finite-shot kernels, S1–S2 (`plans/PREREG_SHOTS.md`) |
| `src/analyze_general_shots.py` | Analysis for the generalization and shot studies |
| `src/figures.py` | Result figures (remedy study and conditioning) |
| `src/fig_data.py` | Data overview figure (features, examples, classes, PCA variance kept) |
| `results/` | Per-split JSON for every study, and the analysis outputs (`*_analysis.txt`) |

## Reproducing the results

Each experiment script skips splits whose JSON already exists. To recompute a study, delete its folder under
`results/`. Run from the repository root:

```
export OMP_NUM_THREADS=1
python src/main.py
python src/confirm.py
python src/remedy.py
python src/general.py
python src/shots.py
```

The scripts run 24 jobs in parallel; on a 32-core machine the five studies take a few hours in total. Rerunning a
stored split gives identical numbers.

Then run the analyses:

```
python src/analyze_main.py
python src/analyze_prereg2.py confirm
python src/analyze_prereg2.py remedy
python src/analyze_general_shots.py general
python src/analyze_general_shots.py shots
python src/figures.py
```

## Notes on the plans

- The files in `plans/` are the dated analysis plans, written before each study was run. They were not registered
  externally.
- Wording that referred to a separate manuscript was replaced with neutral references to the data and grids used
  here. Hypotheses, tests and margins are unchanged.
- Deviations from the plans are listed in the paper.

## License

MIT, see `LICENSE`.
