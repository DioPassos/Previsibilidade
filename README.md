# Previsibilidade

Prototype pipeline for estimating whether Astroshop's HTTP 5xx availability
proxy will fall below 99.95% within the next 30 minutes.

> This is an initial demo, not the application's official 30-day SLO. The
> prototype uses a rolling 60-minute availability window and forecasts the
> next 30 minutes. This is not the application's official SLO. Validate the
> SLO definition with the service owner before operational use.

## Configuration

Set these values at runtime; never bake credentials into the image:

- `DT_TENANT_URL`: Dynatrace SaaS tenant/platform URL.
- `DT_TENANT_API_TOKEN` or `DT_API_TOKEN`: classic token (`dt0c01`) with
  `metrics.read` and `entities.read`, sent using `Api-Token` authorization.
- `DT_SERVICE_NAME_FILTER`: case-insensitive service-name filter (defaults to
  `astroshop`).
- `DT_KUBERNETES_NAMESPACE`: Kubernetes namespace for container metric collection
  (defaults to `astroshop`; confirm this matches the namespace in Dynatrace).
- `DT_LOOKBACK`: collection window (defaults to `24h` for initial history).
- `DT_METRIC_RESOLUTION`: fixed at `5m` for the prototype.
- `RAW_DATA_DIR`, `DATASET_PATH`, `MODEL_PATH`: optional data/artifact paths.
- `RUN_INTERVAL_MINUTES`: periodic collection interval (defaults to 15).
- `DT_INCREMENTAL_LOOKBACK`: lookback on each periodic collection (defaults to
  `1h`; overlapping queries prevent gaps between runs).
- `PREDICTION_THRESHOLD`: alert score threshold (defaults to `0.5`; scores are
  not calibrated probabilities).
- `MAX_FEATURE_AGE_MINUTES`: reject stale inputs (defaults to 15).
- `ALERT_WEBHOOK_URL`: optional HTTPS webhook for risk notifications.

The container and Git repository exclude `.env`, local data, model artifacts,
and logs. Provide secrets using the organization's container secret mechanism
and persist `/Previsibilidade/data` and `/Previsibilidade/models/trained` with
volumes.

## First prototype run

The collector uses the classic Dynatrace Metrics API v2 and Entities API with
`metrics.read` and `entities.read`. It selects only the six Astroshop services
whose HTTP status coverage was confirmed (`frontend`, `cart`, `image-provider`,
`shipping`, `quote`, and `email`) and queries request counts, exact HTTP 5xx
counts, and p95/p99 latency. Services without confirmed HTTP status coverage,
including `flagd` and gRPC services, are excluded from this HTTP SLI. DQL access is not required. Classic service metrics are used because
the API v2 metric IDs `dt.service.*` from the initial suggestion are not
present in this tenant's metric catalog.

1. Configure `DT_TENANT_URL` and `DT_TENANT_API_TOKEN` in the local `.env`.
   The classic token must have `metrics.read` and `entities.read`. The configured
   token has already returned HTTP 200 for both APIs.
2. Collect the first day of history:

   ```bash
   DT_SERVICE_NAME_FILTER=astroshop DT_LOOKBACK=24h python -m collector.collect_pix_data
   ```

3. Build the dataset and inspect its positive-label count:

   ```bash
   python -m feature_engineering.build_dataset
   ```

4. Train and evaluate the chronological baseline:

   ```bash
   python -m training.train_model
   ```

5. Generate a prediction:

   ```bash
   python -m training.predict_model
   ```

The collector queries request, HTTP 5xx, and latency series for the eligible
Astroshop HTTP services. Before saving, it checks that the returned metric
timestamps have a median spacing no coarser than five minutes; if Dynatrace
downsamples a long lookback, collection stops with an explicit error instead
of labeling the data as five-minute. Missing bins remain missing and are not
treated as zero. It also queries Kubernetes workload CPU,
memory, restart, and OOM metrics. Kubernetes responses are saved unchanged in
separate `kubernetes_container_metrics_5m_*.json` files under `data/raw/`, but
the initial model does not use them. Dynatrace confirmed `astroshop` as the
correct namespace for this tenant.

In the live tenant check, a `30d` query returned coarser-than-five-minute
historical points, which cannot fill this prototype's five-minute feature
windows. A fresh `24h` collection returned five-minute data and produced
trainable rows. Confirm the returned timestamps and feature completeness before
using longer lookbacks for training.

The initial model uses 11 features from request volume, HTTP 5xx rate,
latency, and the current 60-minute SLO proxy. Kubernetes metrics are excluded
from the model until they demonstrate measurable value and their event-counter
semantics have been validated further.

The label is positive when a service whose current 60-minute proxy is healthy
falls below the 99.95% target at any point in the next 30 minutes. A complete
60-minute current window and 30 minutes of future data are required for each
training example. Prediction scores are not calibrated probabilities.

Training uses chronological 60%/20%/20% train, validation, and test periods,
with a 120-minute purge before validation and test. The alert threshold is
selected by maximum F1 on validation only when validation contains both label
classes. If not, the preselected default threshold of 0.5 is retained and the
report marks threshold tuning as unsupported. Test results never select the
threshold. The model metadata reports the test precision-recall curve,
fixed-threshold comparisons, feature importance, and grouped positive/alert episodes.
Individual test scores are saved to `data/processed/test_scores.csv`.

A few web-service entities had incomplete metric
coverage and are skipped rather than treating missing data as zero. Training
requires both healthy and proxy-breach labels in the chronological train and
test periods. If the trainer reports too few examples, collect more
representative five-minute history; do not create synthetic production labels.

Once a model has trained, run periodic collection and prediction:

```bash
python -m training.run_periodic
```

The periodic runner does not automatically retrain the model. Review evaluation
metrics before replacing a model. Webhook alerts are sent only when a service
transitions into `risk`; persist the alert-state file with the processed data
volume.

## Prototype limitations

- The label is a 99.95% availability proxy computed over the previous rolling
  hour and checks whether it falls below target during the next 30 minutes. It
  is not the official 30-day rolling SLO.
- Complete five-minute bins are required. Missing bins are unknown, not zero
  traffic or successful requests. Samples where the current proxy is already
  below target are excluded from training.
- The initial model uses request rate, 5xx rate, latency, and proxy
  error-budget features. Kubernetes data is collected for later experiments
  but is not used for this baseline.
- The model emits an uncalibrated score. Do not interpret it as a reliable
  real-world probability until it has been calibrated and validated on more
  independent incidents.
- A model that trains successfully is not necessarily predictive. Use the
  chronological holdout's precision, recall, and PR-AUC, and compare alerts
  against actual Astroshop incidents before presenting it as evidence.
- `python -m training.smoke_test_model` only tests synthetic data and model
  serialization; it is not evidence of predictive quality on Astroshop.
- Webhook payload compatibility (including Microsoft Teams Workflow endpoints)
  must be confirmed with the destination integration before enabling alerts.
