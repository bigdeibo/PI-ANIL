# Physics-informed meta-learning: variant ablation and cross-material extrapolation

R² mean±std, paired splits (rep 0-9), all MAE-initialized.


## 1. Variant ablation (multi task pool, headline 8 tasks + EVOO)


### diesel-CN

| variant                         | 5                | 10              | 20              |
|:--------------------------------|:-----------------|:----------------|:----------------|
| vanilla (route-comparison ANIL) | -0.978±1.195     | -0.071±0.316    | 0.240±0.260     |
| PI: +additivity constraint      | -0.593±1.537     | **0.145±0.162** | 0.152±0.414     |
| PI: +band prior                 | -1.326±2.543     | 0.089±0.269     | **0.260±0.203** |
| PI: both constraints            | **-0.520±1.051** | -0.140±0.426    | 0.018±0.534     |

### diesel-BP50

| variant                         | 5                | 10              | 20              |
|:--------------------------------|:-----------------|:----------------|:----------------|
| vanilla (route-comparison ANIL) | -0.378±0.759     | 0.255±0.328     | **0.539±0.121** |
| PI: +additivity constraint      | **-0.176±0.368** | **0.281±0.315** | 0.498±0.162     |
| PI: +band prior                 | -0.586±0.832     | 0.056±0.387     | 0.410±0.258     |
| PI: both constraints            | -0.269±0.556     | 0.185±0.262     | 0.488±0.211     |

### diesel-D4052

| variant                         | 5               | 10              | 20              |
|:--------------------------------|:----------------|:----------------|:----------------|
| vanilla (route-comparison ANIL) | 0.237±0.312     | 0.473±0.573     | **0.808±0.048** |
| PI: +additivity constraint      | 0.002±0.386     | 0.485±0.215     | 0.679±0.085     |
| PI: +band prior                 | -0.018±0.543    | 0.456±0.325     | 0.738±0.051     |
| PI: both constraints            | **0.244±0.251** | **0.575±0.189** | 0.805±0.057     |

### diesel-FLASH

| variant                         | 5                | 10               | 20              |
|:--------------------------------|:-----------------|:-----------------|:----------------|
| vanilla (route-comparison ANIL) | -1.171±1.375     | -0.383±0.323     | 0.076±0.220     |
| PI: +additivity constraint      | **-0.846±0.576** | **-0.306±0.317** | 0.143±0.110     |
| PI: +band prior                 | -3.188±3.388     | -0.407±0.399     | 0.103±0.162     |
| PI: both constraints            | -1.111±0.968     | -0.404±0.766     | **0.153±0.147** |

### diesel-FREEZE

| variant                         | 5                | 10              | 20              |
|:--------------------------------|:-----------------|:----------------|:----------------|
| vanilla (route-comparison ANIL) | -0.114±0.289     | **0.238±0.244** | **0.508±0.131** |
| PI: +additivity constraint      | -0.168±0.222     | 0.213±0.346     | 0.457±0.183     |
| PI: +band prior                 | -0.870±2.008     | 0.107±0.341     | 0.417±0.165     |
| PI: both constraints            | **-0.106±0.323** | 0.095±0.295     | 0.432±0.175     |

### diesel-TOTAL

| variant                         | 5               | 10              | 20              |
|:--------------------------------|:----------------|:----------------|:----------------|
| vanilla (route-comparison ANIL) | -0.223±1.448    | **0.777±0.052** | **0.843±0.028** |
| PI: +additivity constraint      | 0.566±0.231     | 0.761±0.059     | 0.814±0.055     |
| PI: +band prior                 | 0.537±0.245     | 0.744±0.088     | 0.834±0.054     |
| PI: both constraints            | **0.608±0.069** | 0.733±0.081     | 0.825±0.058     |

### diesel-VISC

| variant                         | 5                | 10              | 20              |
|:--------------------------------|:-----------------|:----------------|:----------------|
| vanilla (route-comparison ANIL) | -1.367±3.137     | **0.312±0.365** | 0.347±0.374     |
| PI: +additivity constraint      | **-0.546±0.878** | 0.141±0.363     | 0.437±0.185     |
| PI: +band prior                 | -1.690±1.926     | 0.160±0.357     | 0.445±0.155     |
| PI: both constraints            | -0.596±0.730     | 0.135±0.375     | **0.473±0.167** |

### gasoline-octane

| variant                         | 5               | 10              | 20              |
|:--------------------------------|:----------------|:----------------|:----------------|
| vanilla (route-comparison ANIL) | 0.247±0.769     | 0.853±0.090     | **0.914±0.027** |
| PI: +additivity constraint      | 0.510±0.263     | 0.839±0.078     | 0.881±0.032     |
| PI: +band prior                 | **0.802±0.161** | **0.876±0.047** | 0.912±0.022     |
| PI: both constraints            | 0.544±0.406     | 0.812±0.106     | 0.912±0.032     |

### evoo-adulteration

| variant                         | 5                | 10               | 20              |
|:--------------------------------|:-----------------|:-----------------|:----------------|
| vanilla (route-comparison ANIL) | -1.071±1.915     | **-0.428±0.702** | 0.146±0.150     |
| PI: +additivity constraint      | **-0.364±0.435** | -0.536±0.929     | **0.176±0.161** |
| PI: +band prior                 | -0.501±0.424     | -0.798±1.000     | -0.058±0.280    |
| PI: both constraints            | -0.633±0.486     | -1.262±1.597     | 0.053±0.161     |


### Headline 8-task means (stability metric: fraction of episodes with R²<-1 per shot level)

| variant                         |   K=5 mean | K=5 failure rate   |   K=10 mean | K=10 failure rate   |   K=20 mean | K=20 failure rate   |
|:--------------------------------|-----------:|:-------------------|------------:|:--------------------|------------:|:--------------------|
| vanilla (route-comparison ANIL) |     -0.468 | 20%                |       0.307 | 0%                  |       0.534 | 0%                  |
| PI: +additivity constraint      |     -0.156 | 10%                |       0.32  | 0%                  |       0.508 | 0%                  |
| PI: +band prior                 |     -0.792 | 25%                |       0.26  | 1%                  |       0.515 | 0%                  |
| PI: both constraints            |     -0.151 | 11%                |       0.249 | 2%                  |       0.513 | 1%                  |


## 2. Cross-material extrapolation (LOMO: target material removed from the meta-training pool)


### Test = gasoline (pool: diesel + corn + EVOO, 12 tasks)

| model                                                              | 5           | 10          | 20          |
|:-------------------------------------------------------------------|:------------|:------------|:------------|
| vanilla ANIL                                                       | 0.394±0.437 | 0.781±0.107 | 0.853±0.050 |
| PI both constraints                                                | 0.386±0.449 | 0.785±0.078 | 0.856±0.042 |
| reference: route-comparison multi pool (incl. gasoline, seen task) | 0.247±0.769 | 0.853±0.090 | 0.914±0.027 |

### Test = 7 diesel properties (pool: gasoline + corn + EVOO, 6 tasks)


#### diesel-CN

| model                                                            | 5            | 10           | 20           |
|:-----------------------------------------------------------------|:-------------|:-------------|:-------------|
| vanilla ANIL                                                     | -1.113±1.719 | -0.285±0.569 | -0.006±0.312 |
| PI both constraints                                              | -0.809±1.557 | -0.021±0.279 | -0.032±0.394 |
| reference: route-comparison multi pool (incl. diesel, seen task) | -0.978±1.195 | -0.071±0.316 | 0.240±0.260  |

#### diesel-BP50

| model                                                            | 5            | 10           | 20          |
|:-----------------------------------------------------------------|:-------------|:-------------|:------------|
| vanilla ANIL                                                     | -1.974±1.657 | -0.485±0.674 | 0.197±0.186 |
| PI both constraints                                              | -1.294±1.400 | -0.294±0.326 | 0.244±0.150 |
| reference: route-comparison multi pool (incl. diesel, seen task) | -0.378±0.759 | 0.255±0.328  | 0.539±0.121 |

#### diesel-D4052

| model                                                            | 5            | 10           | 20          |
|:-----------------------------------------------------------------|:-------------|:-------------|:------------|
| vanilla ANIL                                                     | -0.822±1.019 | -0.044±0.507 | 0.313±0.189 |
| PI both constraints                                              | -1.167±2.475 | 0.330±0.173  | 0.544±0.125 |
| reference: route-comparison multi pool (incl. diesel, seen task) | 0.237±0.312  | 0.473±0.573  | 0.808±0.048 |

#### diesel-FLASH

| model                                                            | 5            | 10           | 20           |
|:-----------------------------------------------------------------|:-------------|:-------------|:-------------|
| vanilla ANIL                                                     | -2.393±1.820 | -0.698±0.480 | -0.065±0.217 |
| PI both constraints                                              | -2.275±1.489 | -0.663±0.853 | 0.042±0.125  |
| reference: route-comparison multi pool (incl. diesel, seen task) | -1.171±1.375 | -0.383±0.323 | 0.076±0.220  |

#### diesel-FREEZE

| model                                                            | 5            | 10           | 20          |
|:-----------------------------------------------------------------|:-------------|:-------------|:------------|
| vanilla ANIL                                                     | -0.988±1.874 | -0.190±0.398 | 0.133±0.195 |
| PI both constraints                                              | -0.802±1.449 | -0.196±0.420 | 0.203±0.194 |
| reference: route-comparison multi pool (incl. diesel, seen task) | -0.114±0.289 | 0.238±0.244  | 0.508±0.131 |

#### diesel-TOTAL

| model                                                            | 5            | 10          | 20          |
|:-----------------------------------------------------------------|:-------------|:------------|:------------|
| vanilla ANIL                                                     | 0.131±0.995  | 0.702±0.112 | 0.767±0.088 |
| PI both constraints                                              | 0.508±0.365  | 0.735±0.071 | 0.780±0.052 |
| reference: route-comparison multi pool (incl. diesel, seen task) | -0.223±1.448 | 0.777±0.052 | 0.843±0.028 |

#### diesel-VISC

| model                                                            | 5            | 10           | 20          |
|:-----------------------------------------------------------------|:-------------|:-------------|:------------|
| vanilla ANIL                                                     | -2.078±1.678 | -0.839±0.913 | 0.023±0.256 |
| PI both constraints                                              | -1.721±1.541 | -0.073±0.479 | 0.215±0.187 |
| reference: route-comparison multi pool (incl. diesel, seen task) | -1.367±3.137 | 0.312±0.365  | 0.347±0.374 |


## 3. Paired Wilcoxon signed-rank tests

| comparison                                  |   n |   median_diff |   mean_diff |   p_value | significant   |
|:--------------------------------------------|----:|--------------:|------------:|----------:|:--------------|
| PI both vs vanilla (headline 8 tasks)       | 240 |        -0.015 |       0.08  |  0.215    | no            |
| PI additivity vs vanilla (headline 8 tasks) | 240 |        -0.018 |       0.1   |  0.295    | no            |
| PI band vs vanilla (headline 8 tasks)       | 240 |        -0.043 |      -0.13  |  0.0014   | yes           |
| PI both vs vanilla (headline, 5-shot only)  |  80 |         0.066 |       0.318 |  0.0607   | no            |
| LOMO gasoline: PI vs vanilla                |  30 |         0.003 |      -0     |  0.839    | no            |
| LOMO diesel: PI vs vanilla                  | 210 |         0.107 |       0.189 |  3.56e-09 | yes           |

(No multiple-comparison correction; the route-comparison vanilla ANIL results and the PI variants share the same MAE initialization and evaluation protocol.)
