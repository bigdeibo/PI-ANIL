# P2a: SimCLR equal-budget (60ep) retraining vs MAE-60 (closing R2-M2)

R2-M2 concern: §3.1 "SimCLR pretraining was systematically weaker than MAE" rested on unequal training budgets (MAE-60ep vs SimCLR-32ep).

This experiment retrains SimCLR to 60ep under an equal budget (two variants cross-validated: cont60 continued from 32ep, scratch60 retrained from scratch with the same seed) and tests whether the conclusion still holds.


Protocol: frozen encoder + ridge-regression probe, 11 tasks × {5,10,20}-shot × 30 reps, seed formula paired with the baselines (bit-identical).

## 1. Per-task median R² (30 reps)


### K=5-shot

| task | MAE-60 | SimCLR-32 | SimCLR-cont60 | SimCLR-scratch60 |
| --- | --- | --- | --- | --- |
| diesel-CN | 0.010 | 0.067 | 0.065 | -0.020 |
| diesel-BP50 | -0.302 | -0.400 | -0.327 | -0.191 |
| diesel-D4052 | 0.024 | -0.161 | -0.123 | -0.136 |
| diesel-FLASH | -0.369 | -0.426 | -0.345 | -0.340 |
| diesel-FREEZE | -0.135 | -0.201 | -0.175 | -0.191 |
| diesel-TOTAL | 0.630 | 0.632 | 0.606 | 0.728 |
| diesel-VISC | -0.688 | -0.934 | -0.741 | -0.835 |
| gasoline-octane | 0.575 | 0.649 | 0.808 | 0.765 |
| corn_m5-protein | -0.044 | -0.103 | -0.060 | -0.087 |
| corn_m5-oil | -0.221 | -0.620 | -0.558 | -0.637 |
| evoo-adulteration | -0.163 | -0.263 | -0.289 | -0.263 |

### K=10-shot

| task | MAE-60 | SimCLR-32 | SimCLR-cont60 | SimCLR-scratch60 |
| --- | --- | --- | --- | --- |
| diesel-CN | 0.093 | 0.156 | 0.170 | -0.061 |
| diesel-BP50 | 0.153 | 0.025 | 0.142 | -0.054 |
| diesel-D4052 | 0.600 | 0.296 | 0.463 | 0.254 |
| diesel-FLASH | -0.089 | -0.140 | -0.072 | -0.079 |
| diesel-FREEZE | 0.020 | 0.023 | 0.075 | -0.135 |
| diesel-TOTAL | 0.785 | 0.735 | 0.794 | 0.827 |
| diesel-VISC | 0.082 | -0.053 | -0.112 | -0.276 |
| gasoline-octane | 0.918 | 0.896 | 0.911 | 0.920 |
| corn_m5-protein | 0.438 | 0.025 | -0.038 | 0.064 |
| corn_m5-oil | 0.302 | 0.081 | 0.132 | -0.075 |
| evoo-adulteration | -0.072 | -0.138 | -0.096 | -0.088 |

### K=20-shot

| task | MAE-60 | SimCLR-32 | SimCLR-cont60 | SimCLR-scratch60 |
| --- | --- | --- | --- | --- |
| diesel-CN | 0.318 | 0.328 | 0.341 | 0.187 |
| diesel-BP50 | 0.611 | 0.486 | 0.507 | 0.408 |
| diesel-D4052 | 0.816 | 0.616 | 0.670 | 0.632 |
| diesel-FLASH | -0.006 | -0.058 | -0.065 | -0.063 |
| diesel-FREEZE | 0.453 | 0.432 | 0.508 | 0.242 |
| diesel-TOTAL | 0.863 | 0.847 | 0.847 | 0.867 |
| diesel-VISC | 0.573 | 0.505 | 0.525 | 0.066 |
| gasoline-octane | 0.934 | 0.925 | 0.949 | 0.952 |
| corn_m5-protein | 0.750 | 0.617 | 0.456 | 0.580 |
| corn_m5-oil | 0.542 | 0.494 | 0.405 | 0.294 |
| evoo-adulteration | 0.232 | -0.030 | -0.201 | -0.019 |


## 2. Headline-8 task summary (mean R² / failure rate = fraction with R²<-1)

| encoder | K=5 R² | K=5 fail | K=10 R² | K=10 fail | K=20 R² | K=20 fail |
| --- | --- | --- | --- | --- | --- | --- |
| MAE-60 | -0.194 | 11% | 0.196 | 5% | 0.486 | 1% |
| SimCLR-32 | -0.508 | 20% | 0.101 | 7% | 0.382 | 2% |
| SimCLR-cont60 | -0.463 | 18% | 0.130 | 4% | 0.412 | 3% |
| SimCLR-scratch60 | -0.430 | 15% | -0.206 | 11% | 0.292 | 4% |


## 3. Paired two-level tests (30 bit-identical reps)

Rep-level Wilcoxon (pooled across task×rep) + task-level sign test (binomial).

| shot | comparison | n | median_diff | Wilcoxon_p | task_level | sign_p |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | MAE-60 vs SimCLR-32 (original unequal 60 vs 32) | 330 | +0.009 | 2.40e-05 | 8/11 pos | 0.227 |
| 5 | MAE-60 vs SimCLR-cont60 (equal budget, continued) | 330 | +0.005 | 3.49e-03 | 8/11 pos | 0.227 |
| 5 | MAE-60 vs SimCLR-scratch60 (equal budget, from scratch) | 330 | +0.003 | 3.11e-03 | 7/11 pos | 0.549 |
| 5 | cont60 vs scratch60 (consistency of the two equal-budget variants) | 330 | +0.002 | 4.07e-01 | 7/11 pos | 0.549 |
| 10 | MAE-60 vs SimCLR-32 (original unequal 60 vs 32) | 330 | +0.065 | 2.42e-11 | 9/11 pos | 0.065 |
| 10 | MAE-60 vs SimCLR-cont60 (equal budget, continued) | 330 | +0.032 | 1.61e-06 | 7/11 pos | 0.549 |
| 10 | MAE-60 vs SimCLR-scratch60 (equal budget, from scratch) | 330 | +0.138 | 1.73e-20 | 8/11 pos | 0.227 |
| 10 | cont60 vs scratch60 (consistency of the two equal-budget variants) | 330 | +0.020 | 8.23e-07 | 7/11 pos | 0.549 |
| 20 | MAE-60 vs SimCLR-32 (original unequal 60 vs 32) | 330 | +0.059 | 5.03e-14 | 10/11 pos | 0.012 |
| 20 | MAE-60 vs SimCLR-cont60 (equal budget, continued) | 330 | +0.056 | 2.12e-13 | 8/11 pos | 0.227 |
| 20 | MAE-60 vs SimCLR-scratch60 (equal budget, from scratch) | 330 | +0.102 | 2.38e-22 | 9/11 pos | 0.065 |
| 20 | cont60 vs scratch60 (consistency of the two equal-budget variants) | 330 | +0.009 | 7.07e-03 | 6/11 pos | 1.000 |

(positive sign = the former's median is better than the latter's; no multiple-comparison correction.)
