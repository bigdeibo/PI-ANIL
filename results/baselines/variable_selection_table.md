# P1b strong baselines: variable-selection PLS (UVE/CARS) vs meta-learning (ANIL/PI-ANIL)

Closes R1-M3 (the chemometrics reviewers' top objection: why not CARS/UVE-PLS?).


Protocol: 30 reps for variable-selection baselines, 10 reps for meta-learning (each method's standard); 
paired statistics use rep 0-9 (the same splits, bit-identical to eval_split). 
Both SNV and variable selection are fitted on the support set (leakage prevention).


## 1. Per-task R² median

PLS/UVE/CARS based on 30 reps; ANIL/PI-ANIL based on 10 reps (each method's standard protocol).


### K=5-shot

| task | PLS | UVE-PLS | CARS-PLS | ANIL(MAE) | PI-ANIL |
| --- | --- | --- | --- | --- | --- |
| diesel-CN | -0.223 | -0.332 | -0.771 | -0.874 | -0.150 |
| diesel-BP50 | -0.863 | -0.886 | -1.426 | -0.127 | -0.306 |
| diesel-D4052 | -0.162 | -0.162 | -1.056 | 0.324 | 0.295 |
| diesel-FLASH | -0.714 | -1.138 | -1.955 | -0.868 | -0.923 |
| diesel-FREEZE | -0.257 | -0.253 | -0.531 | -0.086 | -0.048 |
| diesel-TOTAL | 0.457 | 0.495 | 0.188 | 0.487 | 0.601 |
| diesel-VISC | -1.060 | -1.060 | -1.401 | -0.327 | -0.542 |
| gasoline-octane | 0.816 | 0.695 | 0.559 | 0.591 | 0.698 |
| corn_m5-oil | -0.734 | -0.778 | -1.115 | -1.113 | — |
| corn_m5-protein | -0.178 | -0.171 | -0.325 | -0.088 | — |
| corn_m5-moisture | 0.021 | 0.014 | -0.526 | -0.586 | — |
| corn_m5-starch | -0.698 | -0.636 | -1.411 | -0.513 | — |
| evoo-adulteration | -0.484 | -0.460 | -0.882 | -0.417 | -0.485 |

### K=10-shot

| task | PLS | UVE-PLS | CARS-PLS | ANIL(MAE) | PI-ANIL |
| --- | --- | --- | --- | --- | --- |
| diesel-CN | -0.143 | -0.154 | -0.851 | 0.027 | 0.048 |
| diesel-BP50 | -0.317 | -0.272 | -0.380 | 0.326 | 0.256 |
| diesel-D4052 | 0.486 | 0.117 | 0.196 | 0.704 | 0.582 |
| diesel-FLASH | -0.285 | -0.271 | -0.905 | -0.417 | -0.191 |
| diesel-FREEZE | 0.158 | 0.158 | 0.027 | 0.241 | 0.112 |
| diesel-TOTAL | 0.655 | 0.651 | 0.650 | 0.781 | 0.742 |
| diesel-VISC | -0.068 | -0.099 | -0.189 | 0.447 | 0.197 |
| gasoline-octane | 0.952 | 0.851 | 0.902 | 0.888 | 0.840 |
| corn_m5-oil | 0.226 | 0.144 | 0.045 | 0.066 | — |
| corn_m5-protein | 0.354 | 0.278 | 0.113 | 0.337 | — |
| corn_m5-moisture | 0.428 | 0.420 | 0.297 | 0.075 | — |
| corn_m5-starch | -0.046 | -0.085 | -0.060 | 0.024 | — |
| evoo-adulteration | -0.450 | -0.445 | -0.856 | -0.486 | -0.959 |

### K=20-shot

| task | PLS | UVE-PLS | CARS-PLS | ANIL(MAE) | PI-ANIL |
| --- | --- | --- | --- | --- | --- |
| diesel-CN | 0.090 | 0.142 | -0.027 | 0.326 | 0.062 |
| diesel-BP50 | 0.626 | 0.492 | 0.517 | 0.567 | 0.529 |
| diesel-D4052 | 0.888 | 0.734 | 0.839 | 0.813 | 0.806 |
| diesel-FLASH | -0.090 | -0.080 | -0.317 | 0.154 | 0.191 |
| diesel-FREEZE | 0.473 | 0.356 | 0.320 | 0.518 | 0.473 |
| diesel-TOTAL | 0.807 | 0.838 | 0.827 | 0.850 | 0.832 |
| diesel-VISC | 0.496 | 0.457 | 0.487 | 0.541 | 0.496 |
| gasoline-octane | 0.971 | 0.962 | 0.961 | 0.919 | 0.923 |
| corn_m5-oil | 0.508 | 0.512 | 0.475 | 0.321 | — |
| corn_m5-protein | 0.870 | 0.665 | 0.875 | 0.616 | — |
| corn_m5-moisture | 0.695 | 0.523 | 0.667 | 0.399 | — |
| corn_m5-starch | 0.741 | 0.684 | 0.722 | 0.217 | — |
| evoo-adulteration | 0.092 | 0.018 | 0.125 | 0.107 | 0.080 |


## 2. Headline 8-task summary (R² mean / failure rate = fraction with R²<-1)

| method | K=5 R² | K=5 failure | K=10 R² | K=10 failure | K=20 R² | K=20 failure |
| --- | --- | --- | --- | --- | --- | --- |
| PLS | -0.908 | 26% | -0.293 | 15% | 0.293 | 7% |
| UVE-PLS | -1.070 | 29% | -0.382 | 15% | 0.291 | 5% |
| CARS-PLS | -3.021 | 43% | -0.871 | 24% | 0.110 | 9% |
| ANIL(MAE) | -0.468 | 20% | 0.307 | 0% | 0.534 | 0% |
| PI-ANIL | -0.151 | 11% | 0.249 | 2% | 0.513 | 1% |


## 3. Variable-selection statistics (median selected variable count for UVE/CARS / fallback rate)

Fallback = fewer than 2 variables selected, falling back to full-spectrum plain PLS (n_var ~ full variable count).

| model | shot | task | nvar_med | fallback |
| --- | --- | --- | --- | --- |
| UVE-PLS | 5 | diesel-CN | 401 | 67% |
| UVE-PLS | 5 | diesel-BP50 | 401 | 70% |
| UVE-PLS | 5 | diesel-D4052 | 401 | 70% |
| UVE-PLS | 5 | diesel-FLASH | 401 | 70% |
| UVE-PLS | 5 | diesel-FREEZE | 401 | 80% |
| UVE-PLS | 5 | diesel-TOTAL | 35 | 37% |
| UVE-PLS | 5 | diesel-VISC | 401 | 73% |
| UVE-PLS | 5 | gasoline-octane | 401 | 73% |
| UVE-PLS | 5 | corn_m5-oil | 700 | 77% |
| UVE-PLS | 5 | corn_m5-protein | 700 | 80% |
| UVE-PLS | 5 | corn_m5-moisture | 700 | 77% |
| UVE-PLS | 5 | corn_m5-starch | 700 | 97% |
| UVE-PLS | 5 | evoo-adulteration | 224 | 73% |
| UVE-PLS | 10 | diesel-CN | 228 | 50% |
| UVE-PLS | 10 | diesel-BP50 | 401 | 73% |
| UVE-PLS | 10 | diesel-D4052 | 18 | 43% |
| UVE-PLS | 10 | diesel-FLASH | 401 | 77% |
| UVE-PLS | 10 | diesel-FREEZE | 401 | 60% |
| UVE-PLS | 10 | diesel-TOTAL | 30 | 20% |
| UVE-PLS | 10 | diesel-VISC | 33 | 47% |
| UVE-PLS | 10 | gasoline-octane | 23 | 23% |
| UVE-PLS | 10 | corn_m5-oil | 700 | 57% |
| UVE-PLS | 10 | corn_m5-protein | 366 | 50% |
| UVE-PLS | 10 | corn_m5-moisture | 700 | 83% |
| UVE-PLS | 10 | corn_m5-starch | 700 | 73% |
| UVE-PLS | 10 | evoo-adulteration | 224 | 53% |
| UVE-PLS | 20 | diesel-CN | 47 | 23% |
| UVE-PLS | 20 | diesel-BP50 | 39 | 43% |
| UVE-PLS | 20 | diesel-D4052 | 36 | 33% |
| UVE-PLS | 20 | diesel-FLASH | 401 | 70% |
| UVE-PLS | 20 | diesel-FREEZE | 25 | 40% |
| UVE-PLS | 20 | diesel-TOTAL | 75 | 10% |
| UVE-PLS | 20 | diesel-VISC | 37 | 40% |
| UVE-PLS | 20 | gasoline-octane | 70 | 3% |
| UVE-PLS | 20 | corn_m5-oil | 47 | 37% |
| UVE-PLS | 20 | corn_m5-protein | 30 | 23% |
| UVE-PLS | 20 | corn_m5-moisture | 64 | 47% |
| UVE-PLS | 20 | corn_m5-starch | 700 | 57% |
| UVE-PLS | 20 | evoo-adulteration | 131 | 50% |
| CARS-PLS | 5 | diesel-CN | 9 | 0% |
| CARS-PLS | 5 | diesel-BP50 | 9 | 7% |
| CARS-PLS | 5 | diesel-D4052 | 8 | 0% |
| CARS-PLS | 5 | diesel-FLASH | 7 | 0% |
| CARS-PLS | 5 | diesel-FREEZE | 10 | 10% |
| CARS-PLS | 5 | diesel-TOTAL | 51 | 27% |
| CARS-PLS | 5 | diesel-VISC | 9 | 10% |
| CARS-PLS | 5 | gasoline-octane | 18 | 0% |
| CARS-PLS | 5 | corn_m5-oil | 12 | 3% |
| CARS-PLS | 5 | corn_m5-protein | 12 | 0% |
| CARS-PLS | 5 | corn_m5-moisture | 15 | 0% |
| CARS-PLS | 5 | corn_m5-starch | 10 | 0% |
| CARS-PLS | 5 | evoo-adulteration | 13 | 3% |
| CARS-PLS | 10 | diesel-CN | 6 | 0% |
| CARS-PLS | 10 | diesel-BP50 | 15 | 0% |
| CARS-PLS | 10 | diesel-D4052 | 13 | 3% |
| CARS-PLS | 10 | diesel-FLASH | 7 | 0% |
| CARS-PLS | 10 | diesel-FREEZE | 7 | 3% |
| CARS-PLS | 10 | diesel-TOTAL | 22 | 23% |
| CARS-PLS | 10 | diesel-VISC | 8 | 0% |
| CARS-PLS | 10 | gasoline-octane | 16 | 0% |
| CARS-PLS | 10 | corn_m5-oil | 16 | 0% |
| CARS-PLS | 10 | corn_m5-protein | 10 | 0% |
| CARS-PLS | 10 | corn_m5-moisture | 11 | 0% |
| CARS-PLS | 10 | corn_m5-starch | 14 | 3% |
| CARS-PLS | 10 | evoo-adulteration | 8 | 0% |
| CARS-PLS | 20 | diesel-CN | 11 | 0% |
| CARS-PLS | 20 | diesel-BP50 | 15 | 0% |
| CARS-PLS | 20 | diesel-D4052 | 22 | 0% |
| CARS-PLS | 20 | diesel-FLASH | 6 | 0% |
| CARS-PLS | 20 | diesel-FREEZE | 16 | 0% |
| CARS-PLS | 20 | diesel-TOTAL | 19 | 3% |
| CARS-PLS | 20 | diesel-VISC | 14 | 0% |
| CARS-PLS | 20 | gasoline-octane | 169 | 7% |
| CARS-PLS | 20 | corn_m5-oil | 12 | 0% |
| CARS-PLS | 20 | corn_m5-protein | 17 | 0% |
| CARS-PLS | 20 | corn_m5-moisture | 14 | 0% |
| CARS-PLS | 20 | corn_m5-starch | 20 | 0% |
| CARS-PLS | 20 | evoo-adulteration | 15 | 0% |


## 4. Paired two-level tests (rep 0-9 strict pairing)

Rep-level Wilcoxon (pooled across tasks x reps) + task-level sign test (sign of the median difference across the 13 tasks, binomial).

| shot | comparison | n_rep | median_diff | Wilcoxon_p | task_level | sign_p |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | PI-ANIL vs PLS | 90 | +0.321 | 1.73e-07 | 7/9 positive | 0.180 |
| 5 | PI-ANIL vs UVE-PLS | 90 | +0.420 | 8.25e-08 | 8/9 positive | 0.039 |
| 5 | PI-ANIL vs CARS-PLS | 90 | +0.555 | 1.46e-10 | 9/9 positive | 0.004 |
| 5 | ANIL vs PLS | 130 | +0.146 | 1.25e-02 | 6/13 positive | 1.000 |
| 5 | ANIL vs UVE-PLS | 130 | +0.146 | 4.67e-03 | 8/13 positive | 0.581 |
| 5 | ANIL vs CARS-PLS | 130 | +0.433 | 4.79e-09 | 12/13 positive | 0.003 |
| 10 | PI-ANIL vs PLS | 90 | +0.037 | 1.36e-02 | 6/9 positive | 0.508 |
| 10 | PI-ANIL vs UVE-PLS | 90 | +0.170 | 3.72e-04 | 7/9 positive | 0.180 |
| 10 | PI-ANIL vs CARS-PLS | 90 | +0.298 | 1.67e-07 | 7/9 positive | 0.180 |
| 10 | ANIL vs PLS | 130 | +0.065 | 5.24e-04 | 10/13 positive | 0.092 |
| 10 | ANIL vs UVE-PLS | 130 | +0.210 | 2.07e-08 | 11/13 positive | 0.022 |
| 10 | ANIL vs CARS-PLS | 130 | +0.204 | 1.48e-08 | 11/13 positive | 0.022 |
| 20 | PI-ANIL vs PLS | 90 | +0.024 | 2.99e-02 | 5/9 positive | 1.000 |
| 20 | PI-ANIL vs UVE-PLS | 90 | -0.002 | 5.94e-02 | 4/9 positive | 1.000 |
| 20 | PI-ANIL vs CARS-PLS | 90 | +0.071 | 2.79e-06 | 7/9 positive | 0.180 |
| 20 | ANIL vs PLS | 130 | -0.032 | 4.22e-01 | 7/13 positive | 1.000 |
| 20 | ANIL vs UVE-PLS | 130 | -0.018 | 5.47e-01 | 6/13 positive | 1.000 |
| 20 | ANIL vs CARS-PLS | 130 | +0.014 | 1.41e-01 | 7/13 positive | 1.000 |

(No multiple-comparison correction; task level = on how many tasks PI-ANIL/ANIL beats the baseline by the median.)
