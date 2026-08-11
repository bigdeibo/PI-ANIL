# Meta-learning route comparison (headline tasks: 7 diesel properties + gasoline)

R² mean±std; paired splits; A/B/C are the three routes (frozen/fine-tune, metric-based regression, gradient-based meta-learning).


## diesel-CN

| model                    | 5                | 10              | 20              |
|:-------------------------|:-----------------|:----------------|:----------------|
| PLS                      | -0.615±1.197     | -0.785±1.378    | -0.247±0.901    |
| SVR                      | -0.086±0.129     | **0.014±0.182** | 0.085±0.180     |
| scratch CNN              | -1.125±1.306     | -0.681±1.109    | -0.219±0.332    |
| scratch CNN+Aug          | -0.183±0.524     | -0.071±0.367    | 0.071±0.243     |
| A: frozen MAE + ridge    | -0.122±0.687     | -0.101±0.557    | 0.148±0.518     |
| A: frozen SimCLR + ridge | -0.243±1.225     | -0.222±1.406    | 0.201±0.313     |
| A: MAE fine-tune         | -0.291±0.911     | 0.011±0.376     | -0.202±0.662    |
| B: frozen MAE + kNN      | **-0.052±0.170** | -0.011±0.350    | 0.141±0.108     |
| B: frozen MAE + GP       | -0.133±0.213     | -0.024±0.108    | 0.067±0.111     |
| B: DKL (meta-trained)    | -0.178±0.275     | -0.038±0.078    | -0.091±0.111    |
| C: ANIL (meta-trained)   | -0.978±1.195     | -0.071±0.316    | **0.240±0.260** |
| C: FOMAML (meta-trained) | -2.136±1.932     | -0.881±0.772    | -0.586±0.752    |

## diesel-BP50

| model                    | 5                | 10              | 20              |
|:-------------------------|:-----------------|:----------------|:----------------|
| PLS                      | -1.724±2.240     | -1.195±3.197    | **0.541±0.255** |
| SVR                      | **-0.208±0.287** | 0.014±0.237     | 0.296±0.257     |
| A: frozen MAE + ridge    | -0.534±0.616     | 0.064±0.539     | 0.473±0.372     |
| A: frozen SimCLR + ridge | -0.792±0.984     | -0.013±0.569    | 0.345±0.359     |
| B: frozen MAE + kNN      | -0.316±0.512     | -0.039±0.198    | 0.044±0.182     |
| B: frozen MAE + GP       | -0.215±0.267     | -0.017±0.129    | 0.141±0.164     |
| B: DKL (meta-trained)    | -0.208±0.237     | -0.015±0.144    | 0.039±0.115     |
| C: ANIL (meta-trained)   | -0.378±0.759     | **0.255±0.328** | 0.539±0.121     |
| C: FOMAML (meta-trained) | -3.632±2.770     | -3.003±1.656    | -0.900±0.424    |

## diesel-D4052

| model                    | 5               | 10              | 20              |
|:-------------------------|:----------------|:----------------|:----------------|
| PLS                      | -0.734±1.128    | 0.225±0.779     | **0.846±0.100** |
| SVR                      | -0.549±0.689    | -0.467±0.577    | -0.549±0.732    |
| A: frozen MAE + ridge    | -0.256±1.026    | **0.553±0.200** | 0.770±0.131     |
| A: frozen SimCLR + ridge | -0.559±1.187    | 0.341±0.261     | 0.558±0.196     |
| B: frozen MAE + kNN      | -0.134±0.348    | 0.213±0.166     | 0.350±0.102     |
| B: frozen MAE + GP       | -0.191±0.289    | 0.095±0.201     | 0.350±0.167     |
| B: DKL (meta-trained)    | -0.307±0.413    | 0.057±0.120     | 0.213±0.081     |
| C: ANIL (meta-trained)   | **0.237±0.312** | 0.473±0.573     | 0.808±0.048     |
| C: FOMAML (meta-trained) | -4.681±3.312    | -1.794±1.276    | -1.300±1.397    |

## diesel-FLASH

| model                    | 5                | 10               | 20              |
|:-------------------------|:-----------------|:-----------------|:----------------|
| PLS                      | -2.508±3.542     | -1.689±2.984     | -0.902±1.741    |
| SVR                      | -0.249±0.324     | **-0.071±0.141** | 0.037±0.104     |
| A: frozen MAE + ridge    | -0.629±0.784     | -0.645±1.685     | -0.107±0.433    |
| A: frozen SimCLR + ridge | -0.953±1.600     | -0.507±0.929     | -0.316±1.040    |
| B: frozen MAE + kNN      | -0.452±0.479     | -0.143±0.256     | -0.062±0.206    |
| B: frozen MAE + GP       | **-0.232±0.316** | -0.078±0.125     | -0.024±0.118    |
| B: DKL (meta-trained)    | -0.308±0.348     | -0.082±0.058     | -0.136±0.119    |
| C: ANIL (meta-trained)   | -1.171±1.375     | -0.383±0.323     | **0.076±0.220** |
| C: FOMAML (meta-trained) | -8.357±12.502    | -3.726±4.541     | -1.014±0.648    |

## diesel-FREEZE

| model                    | 5                | 10              | 20              |
|:-------------------------|:-----------------|:----------------|:----------------|
| PLS                      | -0.518±0.711     | -0.094±0.728    | 0.430±0.214     |
| SVR                      | -0.177±0.577     | 0.083±0.196     | 0.390±0.254     |
| A: frozen MAE + ridge    | -0.424±0.605     | 0.006±0.407     | 0.398±0.227     |
| A: frozen SimCLR + ridge | -0.656±1.272     | -0.062±0.445    | 0.347±0.271     |
| B: frozen MAE + kNN      | -0.277±0.704     | -0.100±0.354    | 0.074±0.177     |
| B: frozen MAE + GP       | -0.208±0.625     | 0.024±0.132     | 0.200±0.196     |
| B: DKL (meta-trained)    | -0.425±1.059     | -0.026±0.107    | 0.062±0.153     |
| C: ANIL (meta-trained)   | **-0.114±0.289** | **0.238±0.244** | **0.508±0.131** |
| C: FOMAML (meta-trained) | -4.706±7.922     | -1.660±1.131    | -0.956±0.933    |

## diesel-TOTAL

| model                    | 5               | 10              | 20              |
|:-------------------------|:----------------|:----------------|:----------------|
| PLS                      | 0.213±0.589     | 0.553±0.301     | 0.790±0.081     |
| SVR                      | 0.068±0.324     | 0.415±0.226     | 0.718±0.121     |
| A: frozen MAE + ridge    | **0.581±0.195** | 0.773±0.091     | 0.848±0.070     |
| A: frozen SimCLR + ridge | 0.335±0.757     | 0.608±0.540     | **0.856±0.045** |
| B: frozen MAE + kNN      | 0.403±0.161     | 0.516±0.190     | 0.673±0.067     |
| B: frozen MAE + GP       | -0.053±0.145    | 0.116±0.399     | 0.429±0.154     |
| B: DKL (meta-trained)    | 0.001±0.060     | 0.009±0.180     | 0.276±0.148     |
| C: ANIL (meta-trained)   | -0.223±1.448    | **0.777±0.052** | 0.843±0.028     |
| C: FOMAML (meta-trained) | -3.524±3.131    | -2.300±3.183    | -0.513±0.725    |

## diesel-VISC

| model                    | 5                | 10              | 20              |
|:-------------------------|:-----------------|:----------------|:----------------|
| PLS                      | -1.904±2.465     | -0.300±0.942    | -0.085±1.651    |
| SVR                      | **-0.227±0.351** | 0.082±0.182     | **0.457±0.165** |
| A: frozen MAE + ridge    | -0.609±0.611     | 0.021±0.620     | 0.436±0.393     |
| A: frozen SimCLR + ridge | -1.710±4.107     | -0.123±0.560    | 0.370±0.359     |
| B: frozen MAE + kNN      | -0.411±0.389     | -0.125±0.303    | 0.033±0.202     |
| B: frozen MAE + GP       | -0.278±0.370     | 0.004±0.115     | 0.139±0.144     |
| B: DKL (meta-trained)    | -0.295±0.456     | -0.033±0.100    | 0.064±0.116     |
| C: ANIL (meta-trained)   | -1.367±3.137     | **0.312±0.365** | 0.347±0.374     |
| C: FOMAML (meta-trained) | -6.587±3.594     | -3.087±2.357    | -1.299±1.029    |

## gasoline-octane

| model                    | 5               | 10              | 20              |
|:-------------------------|:----------------|:----------------|:----------------|
| PLS                      | 0.565±0.496     | **0.937±0.054** | **0.967±0.010** |
| SVR                      | 0.149±0.411     | 0.689±0.123     | 0.842±0.077     |
| scratch CNN              | -0.152±0.566    | 0.275±0.349     | 0.842±0.062     |
| scratch CNN+Aug          | 0.504±0.364     | 0.826±0.097     | 0.943±0.010     |
| A: frozen MAE + ridge    | 0.439±0.431     | 0.900±0.053     | 0.923±0.041     |
| A: frozen SimCLR + ridge | 0.518±0.390     | 0.782±0.243     | 0.694±0.969     |
| A: MAE fine-tune         | **0.778±0.176** | 0.916±0.047     | 0.527±0.595     |
| B: frozen MAE + kNN      | 0.038±0.350     | 0.426±0.170     | 0.633±0.129     |
| B: frozen MAE + GP       | -0.185±0.288    | 0.243±0.290     | 0.595±0.124     |
| B: DKL (meta-trained)    | -0.272±0.395    | 0.123±0.067     | 0.250±0.084     |
| C: ANIL (meta-trained)   | 0.247±0.769     | 0.853±0.090     | 0.914±0.027     |
| C: FOMAML (meta-trained) | -0.098±0.651    | 0.076±0.514     | 0.429±0.290     |

## Headline cell win counts (8 tasks x 3 shot levels, counted by largest mean)

|                          |   5 |   10 |   20 |   total |
|:-------------------------|----:|-----:|-----:|--------:|
| C: ANIL (meta-trained)   |   2 |    4 |    3 |       9 |
| SVR                      |   2 |    2 |    1 |       5 |
| PLS                      |   0 |    1 |    3 |       4 |
| A: frozen MAE + ridge    |   1 |    1 |    0 |       2 |
| B: frozen MAE + GP       |   1 |    0 |    0 |       1 |
| B: frozen MAE + kNN      |   1 |    0 |    0 |       1 |
| A: MAE fine-tune         |   1 |    0 |    0 |       1 |
| A: frozen SimCLR + ridge |   0 |    0 |    1 |       1 |


# Ablation: initialization x task source (headline tasks, R² mean)

| config                        |   K=5 headline mean |   K=10 headline mean |   K=20 headline mean |
|:------------------------------|--------------------:|---------------------:|---------------------:|
| DKL MAE + multi-dataset       |              -0.249 |               -0.001 |                0.085 |
| DKL random + multi-dataset    |              -0.265 |               -0.037 |                0.029 |
| DKL MAE + diesel-only         |              -0.273 |               -0.038 |                0.033 |
| ANIL MAE + multi-dataset      |              -0.468 |                0.307 |                0.534 |
| ANIL random + multi-dataset   |              -1.058 |               -0.119 |                0.244 |
| ANIL MAE + diesel-only        |              -0.183 |                0.358 |                0.555 |
| FOMAML MAE + multi-dataset    |              -4.215 |               -2.047 |               -0.767 |
| FOMAML random + multi-dataset |              -4.18  |               -3.408 |               -2.017 |

Note: evaluating source=diesel-only models on the gasoline task is an unseen-task generalization probe (gasoline is absent from their meta-training pool); for multi-dataset models all tasks are seen tasks (sample-level protocol).


# Paired Wilcoxon signed-rank test (headline tasks, paired by rep 0-9)

| comparison                                      |   n |   median_diff |   mean_diff |   p_value | significant   |
|:------------------------------------------------|----:|--------------:|------------:|----------:|:--------------|
| C: ANIL (meta-trained) vs A: frozen MAE + ridge | 240 |         0.027 |      -0.031 |  0.0448   | yes (p<0.05)  |
| C: ANIL (meta-trained) vs B: frozen MAE + GP    | 240 |         0.289 |       0.111 |  1.32e-10 | yes (p<0.05)  |
| C: ANIL (meta-trained) vs B: DKL (meta-trained) | 240 |         0.404 |       0.179 |  2.3e-13  | yes (p<0.05)  |
| ANIL MAE init vs ANIL random init               | 240 |         0.271 |       0.435 |  3.83e-23 | yes (p<0.05)  |
| ANIL multi-dataset vs ANIL diesel-only          | 240 |        -0.056 |      -0.119 |  0.00157  | yes (p<0.05)  |
| DKL MAE init vs DKL random init                 | 240 |         0.013 |       0.036 |  1.77e-11 | yes (p<0.05)  |

(No multiple-comparison correction; p values are for reference. the baseline traditional baselines have no per-rep records and are excluded from paired tests.)
