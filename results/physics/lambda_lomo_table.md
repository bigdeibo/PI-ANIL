# P0 supplementary experiments: lambda sensitivity and LOMO in four directions

R² mean±std, paired splits (rep 0-9), MAE initialization.


## 1. Lambda sensitivity sweep (headline 8 tasks)

| variant | lambda | K=5 mean | K=5 failure rate | K=10 mean | K=20 mean |
|---|---|---|---|---|---|
| vanilla (lambda=0) | 0 | -0.468 | 20% | 0.307 | 0.534 |
| PI additivity | 0.01 | 0.013 | 9% | 0.396 | 0.571 |
| PI additivity | 0.1 | -0.156 | 10% | 0.320 | 0.508 |
| PI additivity | 1.0 | -0.722 | 21% | 0.014 | 0.368 |
| PI band | 0.01 | -0.261 | 15% | 0.446 | 0.610 |
| PI band | 0.1 | -0.792 | 25% | 0.260 | 0.515 |
| PI band | 1.0 | -0.649 | 24% | 0.102 | 0.386 |
| PI both constraints | 0.01/0.01 | -0.457 | 19% | 0.174 | 0.438 |
| PI both constraints | 0.1/0.1 | -0.151 | 11% | 0.249 | 0.513 |

### Paired Wilcoxon vs vanilla (headline 8 tasks, 5-shot only)

| variant lambda | median diff | mean diff | p (cell level) | task-level median diff | p (task level) |
|---|---|---|---|---|---|
| PI additivity lambda=0.01 | +0.164 | +0.481 | 0.0001 | +0.390 | 0.0078 |
| PI additivity lambda=0.1 | -0.002 | +0.312 | 0.4343 | +0.294 | 0.0547 |
| PI additivity lambda=1.0 | -0.130 | -0.254 | 0.0167 | -0.061 | 0.4609 |
| PI band lambda=0.01 | +0.103 | +0.208 | 0.0185 | +0.243 | 0.1484 |
| PI band lambda=0.1 | -0.054 | -0.324 | 0.2141 | -0.289 | 0.4609 |
| PI band lambda=1.0 | -0.205 | -0.180 | 0.0267 | -0.303 | 0.4609 |
| PI both constraints lambda=0.01/0.01 | -0.066 | +0.012 | 0.5142 | -0.009 | 0.9453 |
| PI both constraints lambda=0.1/0.1 | +0.066 | +0.318 | 0.0607 | +0.203 | 0.0078 |

## 2. New LOMO directions: corn and EVOO


### Pool = no_corn (test tasks: corn_m5-moisture, corn_m5-oil, corn_m5-protein, corn_m5-starch)

| task | model | 5 | 10 | 20 |
|---|---|---|---|---|
| corn_m5-moisture | vanilla | -0.821±0.970 | -0.404±0.708 | 0.070±0.263 |
| corn_m5-moisture | PI both constraints | -0.658±0.570 | -0.402±0.671 | 0.180±0.116 |
| corn_m5-moisture | seen-task reference (route-comparison multi pool) | -0.802±0.867 | -0.133±0.569 | 0.364±0.138 |
| corn_m5-oil | vanilla | -0.669±0.583 | 0.124±0.275 | 0.249±0.101 |
| corn_m5-oil | PI both constraints | -0.745±0.391 | 0.096±0.254 | 0.192±0.156 |
| corn_m5-oil | seen-task reference (route-comparison multi pool) | -1.294±0.721 | -0.061±0.446 | 0.287±0.195 |
| corn_m5-protein | vanilla | -0.135±0.633 | 0.123±0.311 | 0.221±0.306 |
| corn_m5-protein | PI both constraints | -0.274±0.973 | 0.139±0.248 | 0.487±0.084 |
| corn_m5-protein | seen-task reference (route-comparison multi pool) | -0.159±0.351 | 0.247±0.421 | 0.603±0.060 |
| corn_m5-starch | vanilla | -0.882±1.048 | -0.254±0.420 | -0.185±0.370 |
| corn_m5-starch | PI both constraints | -0.877±0.993 | -0.114±0.422 | 0.010±0.292 |
| corn_m5-starch | seen-task reference (route-comparison multi pool) | -0.663±0.616 | 0.010±0.214 | 0.159±0.242 |

Paired Wilcoxon (PI - vanilla): n=120, median diff +0.066, mean diff +0.050, p=0.03093; task level: 3/4 positive, median +0.070, p=0.3750


### Pool = no_evoo (test tasks: evoo-adulteration)

| task | model | 5 | 10 | 20 |
|---|---|---|---|---|
| evoo-adulteration | vanilla | -0.558±0.455 | -0.587±0.518 | 0.093±0.063 |
| evoo-adulteration | PI both constraints | -0.462±0.369 | -0.690±0.688 | 0.033±0.166 |
| evoo-adulteration | seen-task reference (route-comparison multi pool) | -1.071±1.817 | -0.428±0.666 | 0.146±0.143 |

Paired Wilcoxon (PI - vanilla): n=30, median diff -0.025, mean diff -0.022, p=0.6266; task level: 0/1 positive, median -0.022, p=1.0000


## 3. LOMO four-direction overview (PI both constraints - vanilla)

| direction (test material) | n (cells) | median diff | mean diff | p (cell level) | task-level positive | p (task level) |
|---|---|---|---|---|---|---|
| diesel | 210 | +0.107 | +0.189 | 3.564e-09 | 7/7 | 0.01562 |
| gasoline | 30 | +0.003 | -0.000 | 0.8394 | 0/1 | 1 |
| corn | 120 | +0.066 | +0.050 | 0.03093 | 3/4 | 0.375 |
| EVOO | 30 | -0.025 | -0.022 | 0.6266 | 0/1 | 1 |

### LOMO diesel: both constraints lambda=0.01 vs lambda=0.1 vs vanilla

| model | K=5 | K=10 | K=20 | vs vanilla p (cell level) | task-level positive |
|---|---|---|---|---|---|
| vanilla | -1.320 | -0.263 | 0.195 | nan | 0/7 |
| PI both lambda=0.1 | -1.080 | -0.026 | 0.285 | 3.564e-09 | 7/7 |
| PI both lambda=0.01 | -1.519 | -0.419 | 0.030 | 0.05059 | 2/7 |
