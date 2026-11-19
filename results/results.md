# Results

Model trained on 1,261,621 trips (2019 H1); 26 reference weeks, 156 monitored weeks.

MAE band from the reference weeks: 2.938 ± 3 × 0.303 (upper 3.847 min). Monitored weeks above it: 21.

## Signals

Degraded weeks are 0.135 of the monitored weeks: the average precision of a signal that ranks weeks at random.

| Signal | Weeks in alert | Precision | Recall | False alarms before the shock | First alert after the shock | Average precision | Spearman with MAE |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| dle_mae | 63 | 0.08 | 0.24 | 0 | 2020-06-08 | 0.16 | -0.03 |
| domain_auc | 126 | 0.17 | 1.00 | 1 | 2020-03-09 | 0.77 | 0.76 |
| psi:prediction | 143 | 0.15 | 1.00 | 0 | 2020-03-16 | 0.12 | -0.09 |
| ks:prediction | 101 | 0.15 | 0.71 | 0 | 2020-03-30 | 0.14 | -0.08 |
| max_feature_psi | 110 | 0.19 | 1.00 | 0 | 2020-03-09 | 0.70 | 0.74 |
| volume_change | 146 | 0.14 | 1.00 | 0 | 2020-03-09 | 0.92 | 0.75 |
| psi:DOLocationID | 81 | 0.25 | 0.95 | 0 | 2020-03-16 | 0.90 | 0.73 |
| psi:PULocationID | 145 | 0.14 | 1.00 | 0 | 2020-03-16 | 0.91 | 0.76 |
| psi:RatecodeID | 122 | 0.16 | 0.95 | 0 | 2020-03-09 | 0.48 | 0.43 |
| psi:VendorID | 51 | 0.04 | 0.10 | 0 | 2020-05-11 | 0.09 | -0.24 |
| psi:do_borough | 83 | 0.24 | 0.95 | 0 | 2020-03-16 | 0.67 | 0.50 |
| psi:hour | 85 | 0.24 | 0.95 | 1 | 2020-03-16 | 0.38 | 0.63 |
| psi:passenger_count | 141 | 0.15 | 1.00 | 0 | 2020-03-09 | 0.81 | 0.46 |
| psi:pu_borough | 137 | 0.14 | 0.90 | 0 | 2020-03-23 | 0.29 | 0.43 |
| psi:trip_distance | 145 | 0.14 | 1.00 | 0 | 2020-03-09 | 0.13 | 0.07 |
| psi:weekday | 14 | 0.29 | 0.19 | 0 | 2020-03-09 | 0.35 | 0.61 |
| js:DOLocationID | 81 | 0.25 | 0.95 | 0 | 2020-03-16 | 0.90 | 0.72 |
| js:PULocationID | 145 | 0.14 | 1.00 | 0 | 2020-03-16 | 0.91 | 0.75 |
| js:RatecodeID | 122 | 0.16 | 0.95 | 0 | 2020-03-09 | 0.49 | 0.45 |
| js:VendorID | 34 | 0.06 | 0.10 | 0 | 2020-05-11 | 0.09 | -0.24 |
| js:do_borough | 80 | 0.25 | 0.95 | 0 | 2020-03-16 | 0.66 | 0.50 |
| js:pu_borough | 130 | 0.15 | 0.90 | 0 | 2020-03-23 | 0.29 | 0.44 |
| ks:hour | 106 | 0.19 | 0.95 | 0 | 2020-03-16 | 0.70 | 0.65 |
| ks:passenger_count | 124 | 0.16 | 0.95 | 0 | 2020-03-09 | 0.91 | 0.57 |
| ks:trip_distance | 127 | 0.13 | 0.76 | 0 | 2020-03-30 | 0.11 | -0.18 |
| ks:weekday | 15 | 0.33 | 0.24 | 0 | 2020-03-09 | 0.39 | 0.57 |

## PSI against the textbook cut-offs (0.1 moderate, 0.25 major)

| Signal | PSI above | Weeks in alert | Precision | Recall | First alert after the shock |
| --- | ---: | ---: | ---: | ---: | --- |
| max_feature_psi | 0.1 | 63 | 0.30 | 0.90 | 2020-03-09 |
| max_feature_psi | 0.25 | 22 | 0.77 | 0.81 | 2020-03-16 |
| psi:DOLocationID | 0.1 | 55 | 0.35 | 0.90 | 2020-03-16 |
| psi:DOLocationID | 0.25 | 14 | 1.00 | 0.67 | 2020-03-23 |
| psi:PULocationID | 0.1 | 59 | 0.32 | 0.90 | 2020-03-16 |
| psi:PULocationID | 0.25 | 15 | 1.00 | 0.71 | 2020-03-23 |
| psi:RatecodeID | 0.1 | 0 | – | 0.00 | – |
| psi:RatecodeID | 0.25 | 0 | – | 0.00 | – |
| psi:VendorID | 0.1 | 1 | 0.00 | 0.00 | 2022-11-14 |
| psi:VendorID | 0.25 | 1 | 0.00 | 0.00 | 2022-11-14 |
| psi:do_borough | 0.1 | 0 | – | 0.00 | – |
| psi:do_borough | 0.25 | 0 | – | 0.00 | – |
| psi:hour | 0.1 | 55 | 0.33 | 0.86 | 2020-03-16 |
| psi:hour | 0.25 | 3 | 0.33 | 0.05 | 2020-04-06 |
| psi:passenger_count | 0.1 | 2 | 0.50 | 0.05 | 2020-03-30 |
| psi:passenger_count | 0.25 | 0 | – | 0.00 | – |
| psi:prediction | 0.1 | 0 | – | 0.00 | – |
| psi:prediction | 0.25 | 0 | – | 0.00 | – |
| psi:pu_borough | 0.1 | 0 | – | 0.00 | – |
| psi:pu_borough | 0.25 | 0 | – | 0.00 | – |
| psi:trip_distance | 0.1 | 0 | – | 0.00 | – |
| psi:trip_distance | 0.25 | 0 | – | 0.00 | – |
| psi:weekday | 0.1 | 5 | 0.20 | 0.05 | 2020-03-09 |
| psi:weekday | 0.25 | 3 | 0.33 | 0.05 | 2020-03-16 |

## Retraining policies (monitored weeks)

| Policy | Retrains | Mean weekly MAE | During the shock (Mar-Jun 2020) | 2022 |
| --- | ---: | ---: | ---: | ---: |
| every week (8-week window) | 155 | 2.672 | 3.017 | 2.954 |
| every 4 weeks (8-week window) | 38 | 2.734 | 3.250 | 3.010 |
| drift-triggered: domain AUC > 0.60 | 13 | 2.855 | 3.186 | 3.116 |
| every 13 weeks (8-week window) | 11 | 2.885 | 3.948 | 3.084 |
| every 13 weeks (52-week window) | 11 | 3.016 | 4.550 | 2.917 |
| error-triggered: MAE > 1.15x at deploy | 1 | 3.089 | 4.049 | 3.127 |
| never retrain | 0 | 3.269 | 4.566 | 3.065 |
