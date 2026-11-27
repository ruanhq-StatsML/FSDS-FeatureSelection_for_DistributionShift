# PermuCATE nuisance ablation (pure concept drift, |S| = 10)

AUC / AP mean (s.e.) over replicates; `mu R2` is the measured out-of-sample R^2 of the outcome model against the true regression. Chance: AUC 0.5, AP = 10/p.

## AUC

| variant | p/n=20 | p/n=100 | p/n=500 |
|:--|---:|---:|---:|
| true mu, true pi | 0.861 (0.034) | 0.788 (0.029) | 0.640 (0.037) |
| true mu, RF pi | 0.838 (0.019) | 0.739 (0.019) | 0.650 (0.022) |
| true mu + noise, nominal R2 0.6 | 0.787 (0.066) | 0.639 (0.061) | 0.570 (0.020) |
| true mu + noise, nominal R2 0.3 | 0.718 (0.047) | 0.658 (0.028) | 0.560 (0.040) |
| true mu + noise, nominal R2 0.1 | 0.676 (0.044) | 0.599 (0.028) | 0.510 (0.029) |
| RF mu, true pi | 0.737 (0.058) | 0.619 (0.020) | 0.520 (0.020) |
| RF mu, RF pi | 0.758 (0.031) | 0.582 (0.012) | 0.520 (0.020) |
| MLP mu, MLP pi | 0.684 (0.038) | 0.581 (0.019) | 0.500 (0.016) |

## AP

| variant | p/n=20 | p/n=100 | p/n=500 |
|:--|---:|---:|---:|
| true mu, true pi | 0.560 (0.049) | 0.345 (0.059) | 0.191 (0.044) |
| true mu, RF pi | 0.526 (0.022) | 0.367 (0.046) | 0.236 (0.037) |
| true mu + noise, nominal R2 0.6 | 0.386 (0.052) | 0.149 (0.025) | 0.107 (0.027) |
| true mu + noise, nominal R2 0.3 | 0.244 (0.051) | 0.051 (0.023) | 0.071 (0.030) |
| true mu + noise, nominal R2 0.1 | 0.217 (0.040) | 0.068 (0.034) | 0.008 (0.006) |
| RF mu, true pi | 0.246 (0.029) | 0.103 (0.004) | 0.005 (0.005) |
| RF mu, RF pi | 0.175 (0.040) | 0.100 (0.033) | 0.010 (0.010) |
| MLP mu, MLP pi | 0.220 (0.044) | 0.108 (0.018) | 0.000 (0.000) |

## mu_r2

| variant | p/n=20 | p/n=100 | p/n=500 |
|:--|---:|---:|---:|
| true mu, true pi | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) |
| true mu, RF pi | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) |
| true mu + noise, nominal R2 0.6 | 0.573 (0.029) | 0.613 (0.034) | 0.613 (0.022) |
| true mu + noise, nominal R2 0.3 | 0.253 (0.050) | 0.323 (0.060) | 0.323 (0.039) |
| true mu + noise, nominal R2 0.1 | 0.039 (0.065) | 0.130 (0.077) | 0.130 (0.050) |
| RF mu, true pi | 0.115 (0.017) | 0.064 (0.011) | 0.063 (0.016) |
| RF mu, RF pi | 0.115 (0.017) | 0.064 (0.011) | 0.063 (0.016) |
| MLP mu, MLP pi | 0.145 (0.025) | 0.079 (0.016) | 0.102 (0.023) |

## Share of the oracle's excess AUC (over 0.5) that is lost to nuisance estimation

| p/n | oracle AUC-0.5 | RF mu+pi | RF mu, true pi | MLP |
|---:|---:|---:|---:|---:|
| 20 | 0.361 | 29% | 34% | 49% |
| 100 | 0.288 | 72% | 59% | 72% |
| 500 | 0.140 | 86% | 86% | 100% |
