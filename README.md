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
  make_figures_paper1.py   all figures
results/        per-cell results, summaries, Wilcoxon tests, chosen hyperparameters
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

### 3. Figures

```bash
python scripts/make_figures_paper1.py
```

## Notes on the protocol

- **Splits:** cells are split into 5 folds × 3 seeds. 20 % of the training cells are used for validation, and test cells are never used for model selection.
- **Inputs:** each prediction uses a causal 30-cycle window. Scalers are fitted on training cells only, and inputs are clipped to ±5 SD.
- **Tuning:** every model gets the same budget, 8 configurations. Selection uses validation cells of two folds of seed 0.
- **Statistics:** per-cell MAE is averaged over seeds. We report 95 % bootstrap CIs over cells and paired Wilcoxon signed-rank tests.
- **Software:** versions are listed in `requirements.txt` and `results/environment_pip_freeze.txt`. Part of the untuned run used PyTorch 2.13; the rankings are unchanged.

## Data sources

- Severson, K. A. et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8
- Wang, F. et al. *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis.* Nature Communications 15, 4332 (2024). https://doi.org/10.1038/s41467-024-48779-z (code: https://github.com/wang-fujin/PINN4SOH)

## Licence

Code: MIT (see `LICENSE`). Third-party data and code keep their original licences.
