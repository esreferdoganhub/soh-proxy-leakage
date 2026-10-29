# Proxy features and honest baselines in battery SOH estimation

Code and results for the manuscript

> **Proxy features and honest baselines in lithium-ion battery state-of-health estimation: a leakage audit across four public datasets** (under preparation).

The study asks two questions:

1. **How much do *proxy features* inflate the apparent accuracy of state-of-health (SOH) estimators?** A proxy feature is an input that is close to a copy of the target, for example the charge capacity of the same cycle or the full discharge curve.
2. **Once leakage is removed, do complex models beat simple ones?** This is checked with cell-held-out evaluation, an equal tuning budget per model, and an audit of a published physics-informed neural network (PINN).

## Main results

| Setting | Result |
|---|---|
| MIT/Severson, 124 LFP cells, gradient boosting | SOH MAE 1.44 % with BMS-measurable signals vs **0.05 %** after adding one proxy feature (charge capacity): **28× apparent improvement** |
| Same, BMS-measurable signals, equal tuning budget | Gradient boosting 1.44 % < Transformer 1.58 % < LSTM 1.80 % < CNN–Transformer 1.87 % (Wilcoxon *p* ≤ 0.01 over cells) |
| Partial discharge curve (≈21 % of capacity) | Recovers 76 % of the full-curve accuracy gain |
| PINN audit (Wang et al., 2024), 4 datasets, 82 test cells | Untuned gradient boosting beats the PINN in 10 of 11 experiments (authors' preprocessing: 0.79 % vs 0.93 %; deployable preprocessing: 0.66 % vs 0.73 %) |
| Larger tuning budget (50 configurations per model) | Ranking unchanged: with BMS signals gradient boosting 1.43 % < Transformer 1.55 % < LSTM 1.72 % |
| Literature survey, 50 highly cited open-access SOH papers | 12 use at least one capacity-equivalent proxy as reported in their methods, 9 more use full constant-current charge time; 22 split cycles of the same cell between training and test; 12 report a simple baseline; 2 of 37 checked papers share code |
| Proxies from the literature on NASA PCoE and CALCE CS2 cells (leave-one-cell-out) | Discharge time to the cutoff voltage lowers MAE 5.4× (NASA, 2.94 % → 0.54 %) and 32× (CALCE, 2.85 % → 0.09 %); errors reported in the literature on these cells (median 0.18–0.61 %) sit in the range of our proxy runs |

All errors are cell-averaged mean absolute errors in % of nominal capacity. Full tables are in `results/`, figures in `figures/`.

## Repository layout

```
src/            HAT-Net (CNN–Transformer) model and configuration
scripts/
  extract_mit_h5.py        MIT .mat batches → per-cycle features (streaming h5py)
  mit_benchmark.py         untuned feature-ladder benchmark (robustness check)
  mit_tuned_benchmark.py   main experiment: equal tuning budget, resumable
  pinn4soh_audit.py        preprocessing conditions A–F with ridge / gradient boosting
  pinn4soh_run.py          runs the authors' PINN unchanged under conditions A and F
  supplementary_analyses.py  K=8 vs K=50, HGB sensitivity in the PINN audit, lifetime-CV analysis
  nasa_calce_extract.py    NASA PCoE / CALCE CS2 raw files → per-cycle features
  nasa_calce_proxy.py      literature proxies on NASA/CALCE (pre-registered plan in results/nasa_calce_replication/PLAN.md)
  mem_watchdog.sh          optional memory guard for long PyTorch runs on macOS
  make_figures_paper1.py   all figures
results/        per-cell results, summaries, Wilcoxon tests, chosen hyperparameters
  lit_proxy_survey/        survey protocol, candidate lists, coding, verification, code check, reported errors
figures/        PDF + PNG figures
data/processed/ (empty; filled by extract_mit_h5.py)
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1. MIT/Severson data (proxy-ladder experiments)

Download the three batch files (CC BY 4.0) from the
[project page](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) (about 8.3 GB):
`2017-05-12_…mat`, `2017-06-30_…mat` and `2018-04-12_…mat`. Then run:

```bash
python scripts/extract_mit_h5.py /path/to/2017-05-12_batchdata_updated_struct_errorcorrect.mat
python scripts/extract_mit_h5.py /path/to/2017-06-30_batchdata_updated_struct_errorcorrect.mat
python scripts/extract_mit_h5.py /path/to/2018-04-12_batchdata_updated_struct_errorcorrect.mat
python scripts/extract_mit_h5.py --merge        # 124 cells, 100 501 cycles
```

Run the experiments:

```bash
python scripts/mit_tuned_benchmark.py --out results/mit_tuned    # ≈15 h on an Apple M2; resumable
python scripts/mit_benchmark.py                                  # untuned robustness check
```

### 2. PINN audit

The PINN code and preprocessed data belong to Wang et al. (2024). They carry no licence and are **not redistributed here**. Clone them at the commit we used:

```bash
git clone https://github.com/wang-fujin/PINN4SOH.git external/PINN4SOH
git -C external/PINN4SOH checkout cc3cc5053caf38f04e0665f7f88cb109144d035e
python scripts/pinn4soh_audit.py   # conditions A–F, ridge / gradient boosting
python scripts/pinn4soh_run.py     # PINN, conditions A and F, 10 repeats
```

TJU test cells are selected with the authors' rule applied to **alphabetically sorted** file names. The original code uses unsorted `os.listdir` order, so TJU splits may differ.

Supplementary analyses (re-create three result files from the outputs above):

```bash
python scripts/mit_tuned_benchmark.py --grid large --K 50 --features ops curve@3.2 --models hgb lstm transformer --out results/mit_tuned_k50
python scripts/supplementary_analyses.py
```

### 3. Literature proxies on NASA and CALCE

Download the NASA PCoE *Battery Data Set* (NASA Prognostics Data Repository, B0005/6/7/18 in `1. BatteryAgingARC-FY08Q4.zip`) and
CALCE CS2_35–CS2_38 (calce.umd.edu). These data are not redistributed here. Point the scripts to them (or place them under `data/raw/`):

```bash
NASA_DIR=/path/to/FY08Q4 CALCE_DIR=/path/to/CALCE_CS2 python scripts/nasa_calce_extract.py
python scripts/nasa_calce_proxy.py
```

### 4. Literature survey

`results/lit_proxy_survey/` holds the protocol, the OpenAlex candidate lists with the exact queries, the coding of 50 papers,
an independent verification of every positive code, the code-availability check and the errors the papers report.
Evidence quotes are limited to 15 words. Full texts of the papers are not included.

### 5. Figures

```bash
python scripts/make_figures_paper1.py
```

## Notes on the protocol

- **Splits:** cells are split into 5 folds × 3 seeds. 20 % of the training cells are used for validation, and test cells are never used for model selection.
- **Inputs:** each prediction uses a causal 30-cycle window. Scalers are fitted on training cells only, and inputs are clipped to ±5 SD.
- **Tuning:** every model gets the same budget, 8 configurations (main run) or 50 configurations drawn from grids of at least 190 combinations (robustness run). Selection uses validation cells of two folds of seed 0.
- **Statistics:** per-cell MAE is averaged over seeds. We report 95 % bootstrap CIs over cells and paired Wilcoxon signed-rank tests.
- **Software:** versions are listed in `requirements.txt` and `results/environment_pip_freeze.txt`. Part of the untuned run used PyTorch 2.13; the rankings are unchanged.

## Data sources

- Severson, K. A. et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8
- Saha, B. & Goebel, K. *Battery Data Set.* NASA Prognostics Data Repository, NASA Ames Research Center (2007).
- He, W., Williard, N., Osterman, M. & Pecht, M. Prognostics of lithium-ion batteries based on Dempster–Shafer theory and the Bayesian Monte Carlo method. *J. Power Sources* 196, 10314–10321 (2011) (CALCE CS2 data). https://doi.org/10.1016/j.jpowsour.2011.08.040
- Wang, F. et al. *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis.* Nature Communications 15, 4332 (2024). https://doi.org/10.1038/s41467-024-48779-z (code: https://github.com/wang-fujin/PINN4SOH)

## Licence

Code: MIT (see `LICENSE`). Third-party data and code keep their original licences.
