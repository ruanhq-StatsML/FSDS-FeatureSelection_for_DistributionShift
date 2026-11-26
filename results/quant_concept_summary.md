# Ultra-high-dim benchmark (pure concept drift (quant alpha-decay formulation), 5% drifting alphas, RF outcome model)

## p/n = 20 (n_exist = n_new = 500, p = 10000, 500 shifted features, chance precision = 0.05)

Grid points completed: 5, replicates per point: 1

### Overall (mean over completed grid points)

| method             |   AUC |    AP |   precision |   lift |   FDR |   recall_2k |   time_sec |   n_grid |
|:-------------------|------:|------:|------------:|-------:|------:|------------:|-----------:|---------:|
| loco_r_risk        | 0.508 | 0.055 |       0.067 |  1.344 | 0.933 |       0.111 |     41.719 |        5 |
| sgshift_l1_spec    | 0.501 | 0.05  |       0.051 |  1.024 | 0.949 |       0.103 |      8.707 |        5 |
| fslnet             | 0.5   | 0.051 |       0.049 |  0.984 | 0.951 |       0.1   |     29.421 |        5 |
| sgshift_l1_misspec | 0.5   | 0.05  |       0.048 |  0.96  | 0.952 |       0.106 |      8.707 |        5 |
| datafix            | 0.5   | 0.05  |       0.052 |  1.048 | 0.948 |       0.106 |      4.475 |        5 |
| rf_domain          | 0.499 | 0.05  |       0.048 |  0.952 | 0.952 |       0.097 |      1.885 |        5 |
| permucate          | 0.498 | 0.051 |       0.051 |  1.024 | 0.949 |       0.099 |      6.742 |        5 |

### Native selection rule (DataFix mask / FSL-Net prob > 0.5)

| method   |   native_n_selected |   native_precision |   native_recall |
|:---------|--------------------:|-------------------:|----------------:|
| datafix  |                 0   |              0     |           0     |
| fslnet   |                77.2 |              0.047 |           0.008 |

PermuCATE: tau-forest splits on 3865 of 10000 features on average (38.6%); the rest get importance exactly 0.

### AUC by rho

| method             |   0.00 |   0.25 |   0.50 |   0.75 |   0.95 |
|:-------------------|-------:|-------:|-------:|-------:|-------:|
| rf_domain          |   0.49 |   0.5  |   0.49 |   0.51 |   0.51 |
| permucate          |   0.5  |   0.51 |   0.49 |   0.49 |   0.5  |
| loco_r_risk        |   0.53 |   0.5  |   0.5  |   0.49 |   0.53 |
| sgshift_l1_spec    |   0.5  |   0.51 |   0.5  |   0.5  |   0.49 |
| sgshift_l1_misspec |   0.5  |   0.5  |   0.5  |   0.5  |   0.5  |
| datafix            |   0.5  |   0.5  |   0.5  |   0.5  |   0.5  |
| fslnet             |   0.49 |   0.51 |   0.51 |   0.5  |   0.49 |

### precision by rho

| method             |   0.00 |   0.25 |   0.50 |   0.75 |   0.95 |
|:-------------------|-------:|-------:|-------:|-------:|-------:|
| rf_domain          |   0.04 |   0.04 |   0.05 |   0.06 |   0.05 |
| permucate          |   0.06 |   0.04 |   0.04 |   0.05 |   0.06 |
| loco_r_risk        |   0.06 |   0.08 |   0.06 |   0.07 |   0.07 |
| sgshift_l1_spec    |   0.06 |   0.06 |   0.05 |   0.04 |   0.04 |
| sgshift_l1_misspec |   0.05 |   0.05 |   0.04 |   0.04 |   0.06 |
| datafix            |   0.04 |   0.07 |   0.05 |   0.04 |   0.06 |
| fslnet             |   0.04 |   0.06 |   0.05 |   0.05 |   0.05 |

## p/n = 100 (n_exist = n_new = 500, p = 50000, 2500 shifted features, chance precision = 0.05)

Grid points completed: 5, replicates per point: 1

### Overall (mean over completed grid points)

| method             |   AUC |    AP |   precision |   lift |   FDR |   recall_2k |   time_sec |   n_grid |
|:-------------------|------:|------:|------------:|-------:|------:|------------:|-----------:|---------:|
| loco_r_risk        | 0.505 | 0.051 |       0.055 |  1.091 | 0.945 |       0.103 |    205.948 |        5 |
| rf_domain          | 0.502 | 0.05  |       0.052 |  1.05  | 0.948 |       0.104 |      4.256 |        5 |
| sgshift_l1_spec    | 0.501 | 0.05  |       0.053 |  1.054 | 0.947 |       0.103 |     81.274 |        5 |
| sgshift_l1_misspec | 0.5   | 0.05  |       0.051 |  1.024 | 0.949 |       0.102 |     81.274 |        5 |
| permucate          | 0.5   | 0.05  |       0.049 |  0.979 | 0.951 |       0.098 |     10.532 |        5 |
| fslnet             | 0.499 | 0.05  |       0.049 |  0.989 | 0.951 |       0.103 |    142.904 |        5 |

### Native selection rule (DataFix mask / FSL-Net prob > 0.5)

| method   |   native_n_selected |   native_precision |   native_recall |
|:---------|--------------------:|-------------------:|----------------:|
| fslnet   |               329.4 |              0.057 |           0.007 |

PermuCATE: tau-forest splits on 4715 of 50000 features on average (9.4%); the rest get importance exactly 0.

### AUC by rho

| method             |   0.00 |   0.25 |   0.50 |   0.75 |   0.95 |
|:-------------------|-------:|-------:|-------:|-------:|-------:|
| rf_domain          |   0.5  |    0.5 |   0.5  |    0.5 |   0.5  |
| permucate          |   0.5  |    0.5 |   0.5  |    0.5 |   0.5  |
| loco_r_risk        |   0.51 |    0.5 |   0.51 |    0.5 |   0.51 |
| sgshift_l1_spec    |   0.5  |    0.5 |   0.5  |    0.5 |   0.5  |
| sgshift_l1_misspec |   0.5  |    0.5 |   0.5  |    0.5 |   0.5  |
| fslnet             |   0.5  |    0.5 |   0.49 |    0.5 |   0.51 |

### precision by rho

| method             |   0.00 |   0.25 |   0.50 |   0.75 |   0.95 |
|:-------------------|-------:|-------:|-------:|-------:|-------:|
| rf_domain          |   0.05 |   0.05 |   0.05 |   0.05 |   0.06 |
| permucate          |   0.05 |   0.05 |   0.05 |   0.04 |   0.05 |
| loco_r_risk        |   0.06 |   0.05 |   0.06 |   0.05 |   0.05 |
| sgshift_l1_spec    |   0.06 |   0.05 |   0.05 |   0.05 |   0.05 |
| sgshift_l1_misspec |   0.06 |   0.06 |   0.05 |   0.05 |   0.05 |
| fslnet             |   0.05 |   0.05 |   0.05 |   0.05 |   0.05 |
