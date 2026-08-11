# P1c: ProtoNet regression (frozen / meta-trained) vs PI-ANIL/ANIL and traditional baselines

Closing the loop on Garzón 2025 (*Eng. Appl. Artif. Intell.*) and its claim that the "ProtoNet family is suitable for few-shot spectroscopic quantification" — ANIL, the method advocated in this project, belongs to the same frozen-encoder family as ProtoNet, and P1c is a direct head-to-head: ProtoNet regression (a Garzón-isomorphic method) vs PI-ANIL/ANIL.


**Two ProtoNet variants** (the only difference = encoder source; same LOO-tau kernel-regression readout):
- `protonet_frozen`: frozen MAE encoder = the MAE counterpart of Garzón's "SSL+PR";
- `protonet_meta`: meta-trained ProtoNet encoder (the proto branch of metatrain.py; same MAE initialization / same episodes / same budget as ANIL/PI-ANIL, only the objective changed to query kernel-regression MSE) = the regression counterpart of Garzón's "meta-trained Prototypical Networks".

Protocol: 30 reps for ProtoNet (frozen-readout convention, same as kNN/GP/PLS); 10 reps for PI-ANIL/ANIL. Bit-identical splits; merge on (task,rep) automatically takes the common reps. Two-level tests: rep-level Wilcoxon + task-level sign test.


## 1. Per-task R² median


### K=5-shot

| task | ProtoNet (meta-trained) | ProtoNet (frozen MAE) | PI-ANIL | ANIL (MAE) | PLS | CARS-PLS | UVE-PLS | kNN (MAE) | GP (MAE) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diesel-CN | -0.026 | -0.004 | -0.150 | -0.874 | -0.223 | -0.771 | -0.332 | -0.010 | -0.045 |
| diesel-BP50 | -0.238 | -0.236 | -0.306 | -0.127 | -0.863 | -1.426 | -0.886 | -0.182 | -0.113 |
| diesel-D4052 | -0.225 | -0.113 | 0.295 | 0.324 | -0.162 | -1.056 | -0.162 | -0.093 | -0.094 |
| diesel-FLASH | -0.277 | -0.204 | -0.923 | -0.868 | -0.714 | -1.955 | -1.138 | -0.274 | -0.124 |
| diesel-FREEZE | -0.119 | -0.108 | -0.048 | -0.086 | -0.257 | -0.531 | -0.253 | -0.088 | -0.103 |
| diesel-TOTAL | 0.451 | 0.544 | 0.601 | 0.487 | 0.457 | 0.188 | 0.495 | 0.452 | -0.021 |
| diesel-VISC | -0.214 | -0.370 | -0.542 | -0.327 | -1.060 | -1.401 | -1.060 | -0.293 | -0.099 |
| gasoline-octane | 0.288 | -0.025 | 0.698 | 0.591 | 0.816 | 0.559 | 0.695 | 0.147 | -0.072 |
| evoo-adulteration | -0.160 | -0.161 | -0.485 | -0.417 | -0.484 | -0.882 | -0.460 | -0.128 | -0.086 |

### K=10-shot

| task | ProtoNet (meta-trained) | ProtoNet (frozen MAE) | PI-ANIL | ANIL (MAE) | PLS | CARS-PLS | UVE-PLS | kNN (MAE) | GP (MAE) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diesel-CN | -0.011 | 0.075 | 0.048 | 0.027 | -0.143 | -0.851 | -0.154 | 0.126 | -0.003 |
| diesel-BP50 | -0.088 | -0.030 | 0.256 | 0.326 | -0.317 | -0.380 | -0.272 | 0.006 | 0.010 |
| diesel-D4052 | 0.199 | 0.254 | 0.582 | 0.704 | 0.486 | 0.196 | 0.117 | 0.236 | 0.105 |
| diesel-FLASH | -0.045 | -0.060 | -0.191 | -0.417 | -0.285 | -0.905 | -0.271 | -0.021 | -0.038 |
| diesel-FREEZE | -0.062 | -0.024 | 0.112 | 0.241 | 0.158 | 0.027 | 0.158 | -0.000 | 0.000 |
| diesel-TOTAL | 0.560 | 0.596 | 0.742 | 0.781 | 0.655 | 0.650 | 0.651 | 0.533 | 0.163 |
| diesel-VISC | -0.108 | -0.072 | 0.197 | 0.447 | -0.068 | -0.189 | -0.099 | -0.038 | 0.016 |
| gasoline-octane | 0.643 | 0.468 | 0.840 | 0.888 | 0.952 | 0.902 | 0.851 | 0.453 | 0.343 |
| evoo-adulteration | -0.164 | -0.052 | -0.959 | -0.486 | -0.450 | -0.856 | -0.445 | 0.000 | -0.035 |

### K=20-shot

| task | ProtoNet (meta-trained) | ProtoNet (frozen MAE) | PI-ANIL | ANIL (MAE) | PLS | CARS-PLS | UVE-PLS | kNN (MAE) | GP (MAE) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diesel-CN | 0.088 | 0.164 | 0.062 | 0.326 | 0.090 | -0.027 | 0.142 | 0.162 | 0.076 |
| diesel-BP50 | 0.067 | 0.058 | 0.529 | 0.567 | 0.626 | 0.517 | 0.492 | 0.089 | 0.155 |
| diesel-D4052 | 0.318 | 0.345 | 0.806 | 0.813 | 0.888 | 0.839 | 0.734 | 0.338 | 0.351 |
| diesel-FLASH | -0.035 | -0.038 | 0.191 | 0.154 | -0.090 | -0.317 | -0.080 | 0.008 | -0.019 |
| diesel-FREEZE | -0.000 | 0.044 | 0.473 | 0.518 | 0.473 | 0.320 | 0.356 | 0.082 | 0.276 |
| diesel-TOTAL | 0.728 | 0.704 | 0.832 | 0.850 | 0.807 | 0.827 | 0.838 | 0.680 | 0.401 |
| diesel-VISC | -0.027 | 0.065 | 0.496 | 0.541 | 0.496 | 0.487 | 0.457 | 0.073 | 0.150 |
| gasoline-octane | 0.740 | 0.672 | 0.923 | 0.919 | 0.971 | 0.961 | 0.962 | 0.648 | 0.608 |
| evoo-adulteration | -0.022 | 0.007 | 0.080 | 0.107 | 0.092 | 0.125 | 0.018 | 0.148 | -0.001 |


## 2. Headline 8-task summary (R² mean / failure rate = fraction with R²<-1)

| method | K=5 R² | K=5 failure | K=10 R² | K=10 failure | K=20 R² | K=20 failure |
| --- | --- | --- | --- | --- | --- | --- |
| ProtoNet (meta-trained) | -0.613 | 8% | -0.254 | 3% | 0.186 | 0% |
| ProtoNet (frozen MAE) | -1.588 | 9% | -0.226 | 2% | 0.225 | 0% |
| PI-ANIL | -0.151 | 11% | 0.249 | 2% | 0.513 | 1% |
| ANIL (MAE) | -0.468 | 20% | 0.307 | 0% | 0.534 | 0% |
| PLS | -0.908 | 26% | -0.293 | 15% | 0.293 | 7% |
| CARS-PLS | -3.021 | 43% | -0.871 | 24% | 0.110 | 9% |
| UVE-PLS | -1.070 | 29% | -0.382 | 15% | 0.291 | 5% |
| kNN (MAE) | -0.150 | 5% | 0.092 | 1% | 0.236 | 0% |
| GP (MAE) | -0.187 | 2% | 0.045 | 0% | 0.237 | 0% |


## 3. Paired two-level tests

(a) **ProtoNet vs the meta-learning family** (rep 0-9, paired with PI-ANIL/ANIL) — the P1c core.

| shot | comparison | n_rep | median_diff | Wilcoxon_p | task_level | sign_p |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | ProtoNet (meta-trained) vs PI-ANIL | 90 | -0.137 | 1.37e-01 | 4/9 positive | 1.000 |
| 5 | ProtoNet (meta-trained) vs ANIL (MAE) | 90 | -0.091 | 7.31e-01 | 3/9 positive | 0.508 |
| 5 | ProtoNet (frozen MAE) vs PI-ANIL | 90 | -0.079 | 4.26e-01 | 5/9 positive | 1.000 |
| 5 | ProtoNet (frozen MAE) vs ANIL (MAE) | 90 | -0.045 | 8.17e-01 | 4/9 positive | 1.000 |
| 5 | ProtoNet (meta-trained) vs ProtoNet (frozen MAE) | 270 | -0.000 | 9.85e-01 | 3/9 positive | 0.508 |
| 10 | ProtoNet (meta-trained) vs PI-ANIL | 90 | -0.220 | 7.97e-04 | 2/9 positive | 0.180 |
| 10 | ProtoNet (meta-trained) vs ANIL (MAE) | 90 | -0.246 | 5.82e-06 | 2/9 positive | 0.180 |
| 10 | ProtoNet (frozen MAE) vs PI-ANIL | 90 | -0.217 | 2.78e-03 | 3/9 positive | 0.508 |
| 10 | ProtoNet (frozen MAE) vs ANIL (MAE) | 90 | -0.256 | 1.57e-04 | 3/9 positive | 0.508 |
| 10 | ProtoNet (meta-trained) vs ProtoNet (frozen MAE) | 270 | -0.006 | 1.98e-02 | 2/9 positive | 0.180 |
| 20 | ProtoNet (meta-trained) vs PI-ANIL | 90 | -0.240 | 1.49e-12 | 1/9 positive | 0.039 |
| 20 | ProtoNet (meta-trained) vs ANIL (MAE) | 90 | -0.276 | 5.18e-13 | 0/9 positive | 0.004 |
| 20 | ProtoNet (frozen MAE) vs PI-ANIL | 90 | -0.270 | 6.22e-12 | 1/9 positive | 0.039 |
| 20 | ProtoNet (frozen MAE) vs ANIL (MAE) | 90 | -0.285 | 4.56e-12 | 0/9 positive | 0.004 |
| 20 | ProtoNet (meta-trained) vs ProtoNet (frozen MAE) | 270 | -0.008 | 3.05e-02 | 4/9 positive | 1.000 |

(Positive sign = the former is better by the median. ProtoNet(meta) vs ProtoNet(frozen) measures the gain of meta-training on the ProtoNet encoder; both use 30 reps.)


(b) **ProtoNet (meta-trained) vs traditional baselines** (rep 0-29, paired with PLS/CARS/UVE/kNN/GP) — context.

| shot | comparison | n_rep | median_diff | Wilcoxon_p | task_level | sign_p |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | ProtoNet (meta-trained) vs PLS | 270 | +0.188 | 1.92e-09 | 6/9 positive | 0.508 |
| 5 | ProtoNet (meta-trained) vs CARS-PLS | 270 | +0.548 | 5.21e-23 | 8/9 positive | 0.039 |
| 5 | ProtoNet (meta-trained) vs UVE-PLS | 270 | +0.197 | 1.30e-09 | 6/9 positive | 0.508 |
| 5 | ProtoNet (meta-trained) vs kNN (MAE) | 270 | -0.009 | 7.12e-01 | 2/9 positive | 0.180 |
| 5 | ProtoNet (meta-trained) vs GP (MAE) | 270 | +0.000 | 1.95e-01 | 3/9 positive | 0.508 |
| 10 | ProtoNet (meta-trained) vs PLS | 270 | -0.058 | 6.74e-01 | 4/9 positive | 1.000 |
| 10 | ProtoNet (meta-trained) vs CARS-PLS | 270 | +0.115 | 7.93e-07 | 6/9 positive | 0.508 |
| 10 | ProtoNet (meta-trained) vs UVE-PLS | 270 | +0.008 | 9.28e-02 | 5/9 positive | 1.000 |
| 10 | ProtoNet (meta-trained) vs kNN (MAE) | 270 | -0.024 | 2.83e-03 | 2/9 positive | 0.180 |
| 10 | ProtoNet (meta-trained) vs GP (MAE) | 270 | +0.002 | 2.46e-01 | 3/9 positive | 0.508 |
| 20 | ProtoNet (meta-trained) vs PLS | 270 | -0.233 | 2.50e-15 | 1/9 positive | 0.039 |
| 20 | ProtoNet (meta-trained) vs CARS-PLS | 270 | -0.175 | 1.49e-06 | 2/9 positive | 0.180 |
| 20 | ProtoNet (meta-trained) vs UVE-PLS | 270 | -0.189 | 3.03e-13 | 1/9 positive | 0.039 |
| 20 | ProtoNet (meta-trained) vs kNN (MAE) | 270 | -0.018 | 1.03e-02 | 2/9 positive | 0.180 |
| 20 | ProtoNet (meta-trained) vs GP (MAE) | 270 | -0.005 | 2.60e-01 | 3/9 positive | 0.508 |

(No multiple-comparison correction.)
