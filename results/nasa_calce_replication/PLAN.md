# Experiment plan: testing the literature's proxy features on NASA and CALCE (pre-registered 2026-10-29, before any data is seen)

## Purpose
The survey showed that the literature uses proxy features; the MIT experiment showed the mechanism. Missing link:
on the datasets the literature actually uses, how much do these exact features inflate accuracy?
Survey basis: NASA/CALCE stratum — 19/25 papers use NASA B0005/6/7/18, 9/25 use CALCE CS2;
most frequent strong features = constant-current (CC) charge time (6 papers), discharge time to cutoff (2–3 papers).

## Data (to download only with user approval, to SSD /Volumes/esref/datasets/)
- NASA PCoE Battery Data Set, zip 210 MB (phm-datasets.s3.amazonaws.com, "5. Battery Data Set"), cells B0005, B0006, B0007, B0018
  (2 A CC discharge, 24 °C, cutoffs 2.7/2.5/2.2/2.5 V; CC 1.5 A charge to 4.2 V then CV to 20 mA).
- CALCE CS2_35/36/37/38, 4 zips, 159 MB total (web.calce.umd.edu/batteries/data/), 1.1 Ah LCO, 1C CC discharge to 2.7 V.
- Secondary (robustness only, if time): other NASA groups at constant conditions.

## Target
SOH = discharge capacity / nominal capacity of the same cycle (NASA nominal 2.0 Ah, CALCE 1.1 Ah). MAE in % of nominal, per cell.

## Feature sets (one ladder per dataset, mirroring the survey)
- F0 realistic: partial-window features available without full cycling — CC charge time inside a mid-voltage window
  (NASA 3.9→4.1 V; CALCE 3.9→4.1 V), mean/max temperature of the cycle, partial discharge time 3.8→3.5 V (TIEVD-style).
- F1 = F0 + full CC charge time from the discharged state (survey category P2c, "conditional").
- F2 = F0 + discharge time from start to the cell's cutoff voltage (P2, strong; = Q/I under CC).
- F3 = F0 + same-cycle charge capacity (P1) — CALCE has it directly; NASA via Coulomb counting of the charge record.

## Protocols (crossed with every feature set)
- S1 within-cell chronological split (first 70 % cycles train, rest test) — the most common in the survey.
- S2 leave-one-cell-out (cell-held-out).
- N1 train-only normalization; N2 per-cell whole-life min–max normalization (the PINN4SOH pattern), each crossed with S2.

## Models
Linear regression (ridge, α by inner leave-one-cell-out), gradient boosting (HGB, fixed defaults + the 4 configs used in the PINN audit),
and a small LSTM only if time allows. No test data in any selection step.

## Hypotheses (stated before running)
H1 F2 and F3 reduce MAE by a large factor (≥5×) relative to F0 under S2, on both datasets.
H2 F1 (full CC charge time) reduces MAE substantially, less than F2/F3.
H3 S1 (within-cell) gives lower MAE than S2 for every feature set.
H4 N2 lowers MAE vs N1 when lifetimes are similar (NASA B5/6/7) — exploratory, since n cells is tiny.

## Statistics and honesty rules
- Only 4 cells per dataset → no cell-level significance tests; report per-cell MAE for every cell, and bootstrap over cycles within cell only as descriptive.
- Report all results, including hypotheses that fail. Any change to this plan after seeing results is logged below with a reason.
- Compare our proxy MAEs with the range of MAEs the surveyed papers report on the same cells (extracted from the coded texts) — descriptive only.

## Deliverables
results/nasa_calce_replication/{per_cell.csv, summary.csv, figure}, one new subsection in Results (or Supplementary if the
figure/table count would exceed 10), and one sentence in the abstract only if H1 holds on both datasets.

## Change log
(empty)
- 2026-10-29 (after inspecting raw data STRUCTURE only, before any feature/target analysis or model run):
  the partial discharge window (3.8→3.5 V) is moved out of F0 into its own rung "F2p", because it requires a constant-current
  discharge starting from full charge — the same reasoning as the MIT partial-curve rungs; its share of capacity will be reported.
  F0 is therefore charge-side only: CC time in the 3.9→4.1 V charge window + mean/max temperature during that window.
  Pairing: features of the charge that immediately PRECEDES discharge k are paired with the capacity of discharge k (NASA order is charge→discharge).
