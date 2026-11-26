# Pure concept drift, p/n=20, rho=0.5 (grid idx 10): selection AUC vs signal strength (snr, drift_ratio)

| method             |   snr1_drift2 |   snr3_drift8 |   snr10_drift30 |
|:-------------------|--------------:|--------------:|----------------:|
| rf_domain          |         0.493 |         0.493 |           0.493 |
| permucate          |         0.486 |         0.509 |           0.511 |
| loco_r_risk        |         0.495 |         0.515 |           0.527 |
| sgshift_l1_spec    |         0.499 |         0.509 |           0.51  |
| sgshift_l1_misspec |         0.498 |         0.508 |           0.505 |
| datafix            |         0.5   |         0.5   |           0.5   |
| fslnet             |         0.51  |         0.51  |           0.51  |
