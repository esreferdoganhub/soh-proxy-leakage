# Results vs pre-registered hypotheses (2026-10-29)
Honest protocol = S2 (leave-one-cell-out) + N1 (train-only normalization). Cell-averaged MAE, % of nominal. 4 cells per dataset → descriptive only.

| | NASA best | CALCE best |
|---|---|---|
| F0 realistic (charge window) | 2.94 (HGB) | 2.85 (HGB) |
| F1 + full CC charge time | 2.19 (ridge) | 0.89 (ridge) |
| F2p + "partial" discharge 3.8→3.5 V (covers 39–57 % / ~80 % of capacity) | 2.79 (HGB) | 1.22 (HGB) |
| F2 + discharge time to cutoff | 0.54 (ridge) | 0.09 (ridge) |
| F3 + charge capacity | 1.35 (HGB) | 0.53 (ridge) |
| training mean | 8.56 | 11.94 |

- H1 (F2 and F3 ≥5× lower than F0): F2 SUPPORTED on both (5.4× NASA, 32× CALCE). F3 supported on CALCE (5.4×), NOT on NASA (2.2×).
- H2 (F1 lowers error, less than F2/F3): SUPPORTED (NASA 1.3×, CALCE 3.2×).
- H3 (within-cell split lower error than cell-held-out): NOT SUPPORTED. The chronological within-cell split (first 70 % train) is an
  extrapolation to lower SOH; HGB cannot extrapolate (CALCE 15.5 vs 2.9). Linear models with a proxy still extrapolate well (CALCE F2 ridge 0.62, NASA 0.23).
- H4 (whole-life normalization helps on NASA): NOT SUPPORTED / mixed (ridge F0 better 3.52 vs 5.47, HGB worse 4.10 vs 2.94).
  With proxies, per-cell normalization removes most of the proxy gain (NASA F2 ridge 3.11 vs 0.54) because it destroys the absolute scale.
- HGB sensitivity (4 configs): S2/N1 results change by ≤0.05 percentage points.
Not done yet: comparison with MAEs reported by the surveyed papers on the same cells.
