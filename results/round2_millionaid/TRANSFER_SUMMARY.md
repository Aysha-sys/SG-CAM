# SG-CAM transfer and robustness results

Generated Sun Oct 25 13:16:09 2026 | Tesla T4 | runtime 44.8 min

Target classes: ['dry field', 'greenhouse', 'paddy field', 'terraced field'] | counts: [366, 159, 1033, 674] | split seeds: [0, 1, 2, 3, 4] | PatternNet checkpoint seeds B: [0, 1, 2, 3, 4], C: [0, 1, 2, 3, 4], E: [0, 1, 2]

## Million-AID subset

| Class | Total | Train | Validation | Test |
|---|---|---|---|---|
| dry field | 366 | 220 | 73 | 73 |
| greenhouse | 159 | 95 | 32 | 32 |
| paddy field | 1033 | 620 | 206 | 207 |
| terraced field | 674 | 404 | 135 | 135 |

## Linear probe

| Feature extractor (frozen) | Accuracy (%) | Balanced accuracy (%) | Macro F1 (%) | Macro AUC (OvR) | Runs (n) |
|---|---|---|---|---|---|
| A: ImageNet MobileNetV2 | 99.02 ± 0.56 | 98.76 ± 0.78 | 98.63 ± 0.73 | 0.9996 ± 0.0002 | 5 |
| B: PatternNet MobileNetV2 (no SG-CAM) | 98.41 ± 0.48 | 98.08 ± 0.89 | 97.95 ± 0.85 | 0.9992 ± 0.0008 | 25 |
| C: PatternNet SG-CAM (proposed) | 98.63 ± 0.61 | 98.02 ± 0.92 | 98.23 ± 0.88 | 0.9994 ± 0.0007 | 25 |
| D: SegFormer-B2 features only | 98.12 ± 0.46 | 97.50 ± 0.99 | 97.44 ± 0.80 | 0.9992 ± 0.0002 | 5 |
| E: PatternNet concatenation fusion | 98.76 ± 0.59 | 98.22 ± 1.14 | 98.30 ± 0.93 | 0.9995 ± 0.0004 | 15 |

## Paired tests (C vs B)

| Comparison | Metric | Pairs (n) | Mean difference (pp) | Wilcoxon p | Paired t-test p |
|---|---|---|---|---|---|
| C (SG-CAM) − B (no SG-CAM) | accuracy | 25 | +0.22 | 0.1 | 0.131 |
| C (SG-CAM) − E (concatenation) | accuracy | 15 | -0.24 | 0.28 | 0.243 |
| C (SG-CAM) − B (no SG-CAM) | balanced_accuracy | 25 | -0.06 | 0.731 | 0.792 |
| C (SG-CAM) − E (concatenation) | balanced_accuracy | 15 | -0.41 | 0.229 | 0.221 |
| C (SG-CAM) − B (no SG-CAM) | macro_f1 | 25 | +0.27 | 0.182 | 0.197 |
| C (SG-CAM) − E (concatenation) | macro_f1 | 15 | -0.27 | 0.359 | 0.363 |

## Few-shot balanced accuracy (%)

| extractor | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| A: ImageNet MobileNetV2 | 72.9 | 87.54 | 90.29 | 91.44 |
| B: PatternNet MobileNetV2 (no SG-CAM) | 66.27 | 81.36 | 84.23 | 85.41 |
| C: PatternNet SG-CAM (proposed) | 68.4 | 81.84 | 84.57 | 85.61 |
| D: SegFormer-B2 features only | 62.94 | 82.71 | 86.18 | 87.13 |
| E: PatternNet concatenation fusion | 67.26 | 82.15 | 85.71 | 87.43 |

## Robustness

| Degradation | Level | B: no SG-CAM (%) | C: SG-CAM (%) | E: concatenation fusion (%) | Δ C−B (pp) | Δ C−E (pp) |
|---|---|---|---|---|---|---|
| Clean | – | 99.67 ± 0.03 | 99.57 ± 0.06 | 99.68 ± 0.02 | -0.11 | -0.11 |
| Gaussian blur (σ) | 1 | 94.89 ± 2.60 | 93.71 ± 2.46 | 94.68 ± 0.52 | -1.18 | -0.97 |
| Gaussian blur (σ) | 2 | 73.70 ± 5.11 | 72.39 ± 3.37 | 78.82 ± 2.00 | -1.31 | -6.43 |
| Gaussian blur (σ) | 3 | 45.14 ± 9.10 | 51.94 ± 10.22 | 59.44 ± 4.24 | +6.81 | -7.50 |
| Gaussian noise (σ) | 0.04 | 66.31 ± 12.68 | 78.60 ± 7.35 | 73.46 ± 9.15 | +12.29 | +5.14 |
| Gaussian noise (σ) | 0.08 | 24.64 ± 9.72 | 39.81 ± 16.66 | 28.64 ± 6.16 | +15.16 | +11.17 |
| Gaussian noise (σ) | 0.12 | 12.80 ± 4.28 | 22.61 ± 12.26 | 11.24 ± 2.96 | +9.81 | +11.37 |
| Brightness (factor) | 0.5 | 98.27 ± 0.55 | 98.57 ± 0.26 | 98.88 ± 0.20 | +0.30 | -0.31 |
| Brightness (factor) | 0.75 | 99.49 ± 0.10 | 99.38 ± 0.06 | 99.56 ± 0.04 | -0.11 | -0.19 |
| Brightness (factor) | 1.5 | 99.21 ± 0.07 | 99.18 ± 0.12 | 99.37 ± 0.05 | -0.02 | -0.19 |
| Haze (α) | 0.2 | 99.34 ± 0.18 | 99.13 ± 0.09 | 99.45 ± 0.07 | -0.21 | -0.31 |
| Haze (α) | 0.4 | 91.79 ± 1.92 | 94.52 ± 1.01 | 95.00 ± 2.20 | +2.73 | -0.48 |
| Haze (α) | 0.6 | 65.57 ± 4.88 | 81.13 ± 4.33 | 76.99 ± 4.93 | +15.56 | +4.14 |
| JPEG (quality) | 30 | 99.14 ± 0.14 | 98.90 ± 0.04 | 99.16 ± 0.08 | -0.24 | -0.25 |
| JPEG (quality) | 15 | 96.96 ± 0.44 | 96.32 ± 0.49 | 97.09 ± 0.28 | -0.64 | -0.78 |
| JPEG (quality) | 5 | 62.73 ± 6.69 | 63.87 ± 3.76 | 62.57 ± 7.53 | +1.14 | +1.29 |
| Down-sampling (×) | 2 | 96.03 ± 2.16 | 95.18 ± 1.99 | 96.19 ± 0.72 | -0.85 | -1.01 |
| Down-sampling (×) | 4 | 73.80 ± 6.55 | 72.69 ± 3.34 | 79.93 ± 0.84 | -1.11 | -7.24 |
| Down-sampling (×) | 8 | 26.29 ± 8.46 | 34.56 ± 9.19 | 39.78 ± 3.00 | +8.28 | -5.22 |

## Robustness summary

| Model | Mean accuracy over all degradations (%) | Seeds (n) |
|---|---|---|
| B: no SG-CAM | 73.67 ± 2.12 | 5 |
| C: SG-CAM | 77.36 ± 1.46 | 5 |
| E: concatenation fusion | 77.24 ± 0.24 | 3 |
| Δ C−B (pp), seed-paired | +3.69 (paired t-test p = 0.0017) | 5 |
| Δ C−E (pp), seed-paired | +0.56 (paired t-test p = 0.488) | 3 |

