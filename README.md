# nyc-taxi-model-monitoring

[![CI](https://github.com/hoang-stack/nyc-taxi-model-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/hoang-stack/nyc-taxi-model-monitoring/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**What would have told you that your model broke?** This repo trains a trip-duration (ETA)
model on NYC yellow-taxi trips from early 2019, freezes it, and replays three years of real
traffic through it, including the COVID-19 shock of March 2020. Every week it logs the model's
real error next to the usual monitoring signals: feature drift (PSI, KS, Jensen-Shannon),
multivariate drift (a domain classifier), prediction drift, and a label-free estimate of the
error (direct loss estimation). It then asks which of those signals would have told an on-call
engineer that the error had changed, early and without false alarms, and replays seven
retraining policies on the same stream.

The same monitoring runs in production form: a FastAPI service with schema validation,
Prometheus metrics (including live PSI per feature), prediction and feedback logs, a batch
monitoring job that exits non-zero on alerts, a file-based model registry with atomic
promote and rollback, and a Docker image that CI builds and smoke-tests.

<!-- RESULTS -->

## Study design

- **Data:** [NYC TLC yellow-taxi trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
  January 2019 to December 2022, a 3% uniform random sample of every month (5.3 million
  trips). The fraction is fixed, not the count, so traffic volume stays visible and a week
  that spans two months is not distorted. Raw files are deleted after sampling.
- **Model:** LightGBM on log trip duration from what is known when a trip is requested:
  pickup and drop-off zone and borough, hour, weekday, planned distance, rate code, vendor,
  passenger count. Fare and drop-off time are not features.
- **Periods, by pickup week:** train 2019 H1 (once, then frozen); **reference** 2019 H2,
  ordinary weeks with labels where every alert threshold and the error-estimation model are
  calibrated; **monitored** 30 Dec 2019 to 25 Dec 2022 (156 weeks).
- **Thresholds:** every signal alerts when it leaves mean + 3 sd of its own values over the
  26 reference weeks, the rule NannyML and most monitoring tools use. None is tuned on the
  monitored period.
- **Ground truth:** a monitored week is *degraded* when the realised MAE leaves its own
  reference band in the same way. Each signal is scored by precision and recall against
  these weeks, by false alarms in the 10 ordinary weeks before the shock, and by how early
  it fired after it.
- **Data quality:** ten checks (nulls, non-positive or sub-minute durations, trips over 3 h,
  zero or > 100-mile distances, speeds over 80 mph, unknown zones, pickups outside the file's
  month). Failing rows are excluded from training and scoring, and the failure rate per week is
  itself monitored.

<!-- ANALYSIS -->

## Run it

```bash
git clone https://github.com/hoang-stack/nyc-taxi-model-monitoring.git
cd nyc-taxi-model-monitoring
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # drift statistics vs scipy/sklearn, service, registry, monitor

taximon download                # 48 monthly files, ~2.5 GB streamed, 170 MB kept
taximon study                   # frozen model + every weekly signal  -> results/weekly.csv
taximon policies                # retraining replay (slowest step)    -> results/policies_weekly.csv
taximon report                  # figures + results/results.md
```

### Serve and monitor

```bash
taximon train --start 2022-07-04 --end 2022-12-19 --version v2022h2 --promote
taximon serve                   # or: docker compose up  (API on :8000, Prometheus on :9090)

curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"pickup_datetime": "2022-12-01T08:30", "pu_location_id": 132,
       "do_location_id": 230, "trip_distance": 17.2}'
# {"request_id": "...", "eta_minutes": ..., "model_version": "v2022h2"}

curl -X POST localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"request_id": "...", "actual_minutes": 52.0}'

taximon monitor                 # drift + realised MAE over logs/, exit code 1 on alert
taximon versions                # registry; `taximon promote <version>` rolls back
```

The image installs only what serving needs. The model is mounted, not baked in, so promoting
or rolling back a model does not need a rebuild. `python -m taximon.synthetic` writes a small
demo model trained on synthetic trips, which CI uses to build the image and smoke-test it.
[deploy/alerts.yml](deploy/alerts.yml) has Prometheus alert rules for feature drift,
prediction drift and latency.

## Project layout

```text
src/taximon/
├── download.py      monthly TLC files -> fixed-fraction samples
├── data.py          loading, ten data-quality checks, features, weekly batches
├── model.py         the monitored LightGBM ETA model (save / load)
├── drift.py         PSI, KS, Jensen-Shannon, domain-classifier AUC, AUC, thresholds
├── performance.py   realised MAE / MAPE / bias, direct loss estimation (DLE)
├── study.py         the frozen-model study and the policy replay driver
├── evaluate.py      scoring signals against real error changes
├── policies.py      retraining policies replayed on the weekly stream
├── report.py        figures and results/results.md
├── profile.py       reference profile shipped with each model (for live PSI)
├── registry.py      models/<version> + atomic `current` symlink
├── service.py       FastAPI app: /predict /feedback /metrics /health, JSONL logs
├── monitor_logs.py  batch monitoring job over the service logs
├── synthetic.py     synthetic trips for tests and the demo model
└── cli.py           `taximon ...`
deploy/              Prometheus scrape config and alert rules
```

<!-- LIMITATIONS -->

## References

- NYC Taxi & Limousine Commission. TLC Trip Record Data.
- Rabanser, S., Günnemann, S., Lipton, Z. (2019). Failing loudly: an empirical study of methods for detecting dataset shift. *NeurIPS.*
- Lipton, Z., Wang, Y.-X., Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML.*
- Gama, J. et al. (2014). A survey on concept drift adaptation. *ACM Computing Surveys.*
- NannyML documentation: direct loss estimation and confidence-based performance estimation.
- Yurdakul, B. (2018). Statistical properties of population stability index. *Western Michigan University.*
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS.*
