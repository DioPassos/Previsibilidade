import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from collector.collect_pix_data import SERVICE_NAME_ALIASES, SERVICE_WORKLOADS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
INTERVAL = "5min"
WINDOW_60M = 12
HORIZON_30M = 6
HORIZON_MINUTES = 30
SLO_TARGET = 0.9995
FEATURE_COLUMNS = [
    "request_rate_mean_60m",
    "request_rate_delta_30m",
    "error_rate_5xx_mean_60m",
    "error_rate_5xx_max_60m",
    "error_rate_5xx_delta_30m",
    "p95_latency_ms_mean_60m",
    "p95_latency_ms_max_60m",
    "p99_latency_ms_mean_60m",
    "p99_latency_ms_max_60m",
    "slo_current",
    "slo_error_budget_remaining_pct",
]
KUBERNETES_METRIC_KEYS = (
    "builtin:kubernetes.workload.cpu_usage",
    "builtin:kubernetes.workload.requests_cpu",
    "builtin:kubernetes.workload.limits_cpu",
    "builtin:kubernetes.workload.memory_working_set",
    "builtin:kubernetes.workload.requests_memory",
    "builtin:kubernetes.workload.limits_memory",
    "builtin:kubernetes.container.restarts",
    "builtin:kubernetes.container.oom_kills",
)
COUNT_COLUMNS = ["total", "errors5xx"]
RECORD_KEY_ALIASES = {
    "bin_timestamp": ("bin_timestamp", "timestamp", "binTimestamp"),
    "service": ("dt.service.name", "service", "service_name"),
    "total": ("total", "total_requests"),
    "errors5xx": ("errors5xx", "errors_5xx", "errors5xx_count"),
}
OPTIONAL_METRIC_ALIASES = {
    "p95_latency_ms": ("p95_latency_ms", "p95_ms"),
    "p99_latency_ms": ("p99_latency_ms", "p99_ms"),
}


def _first(record: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in record:
            return record[key]
    return None


def load_request_observations(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    files = sorted(raw_path.glob("*_request_counts_5m_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No request_counts_5m raw files found in {raw_path}; run the collector first"
        )

    observations = []
    for path in files:
        try:
            payload = json.loads(path.read_text())
            records = payload["response"]["result"]["records"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid request-count response file: {path.name}") from error

        sampled = (
            payload.get("response", {})
            .get("metadata", {})
            .get("grail", {})
            .get("sampled", False)
        )
        if sampled:
            raise ValueError(f"Sampled Dynatrace data is not suitable for labels: {path.name}")
        if not isinstance(records, list):
            raise ValueError(f"Expected result.records to be a list in {path.name}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Unexpected record format in {path.name}")
            timestamp = _first(record, RECORD_KEY_ALIASES["bin_timestamp"])
            service = _first(record, RECORD_KEY_ALIASES["service"])
            total = _first(record, RECORD_KEY_ALIASES["total"])
            errors = _first(record, RECORD_KEY_ALIASES["errors5xx"])
            if timestamp is None or service is None or total is None or errors is None:
                raise ValueError(
                    f"Missing timestamp, service, total, or errors5xx in {path.name}"
                )
            row = {
                "timestamp": pd.to_datetime(timestamp, utc=True, errors="coerce"),
                "service": SERVICE_NAME_ALIASES.get(str(service), str(service)),
                "total": pd.to_numeric(total, errors="coerce"),
                "errors5xx": pd.to_numeric(errors, errors="coerce"),
            }
            for output_column, aliases in OPTIONAL_METRIC_ALIASES.items():
                value = _first(record, aliases)
                row[output_column] = (
                    pd.to_numeric(value, errors="coerce") if value is not None else np.nan
                )
            observations.append(row)

    frame = pd.DataFrame(observations)
    frame = frame.dropna(subset=["timestamp", "service", "total", "errors5xx"])
    if frame.empty:
        raise ValueError("Dynatrace returned no usable request-count observations")
    if (frame["total"] < 0).any() or (frame["errors5xx"] < 0).any():
        raise ValueError("Request and error counts cannot be negative")
    if (frame["errors5xx"] > frame["total"]).any():
        raise ValueError("5xx count exceeds total known-status requests")

    frame["timestamp"] = frame["timestamp"].dt.floor(INTERVAL)
    frame = frame.sort_values("timestamp")
    return frame.drop_duplicates(["service", "timestamp"], keep="last").reset_index(drop=True)


def load_kubernetes_observations(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    files = sorted(raw_path.glob("kubernetes_container_metrics_5m_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No Kubernetes metric files found in {raw_path}; run the collector first"
        )

    latest_points = {}
    for path in files:
        try:
            payload = json.loads(path.read_text())
            metric_responses = payload["metric_responses"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid Kubernetes metric file: {path.name}") from error
        if payload.get("namespace") != "astroshop":
            raise ValueError(
                f"Expected namespace 'astroshop' in {path.name}, "
                f"got {payload.get('namespace')!r}"
            )

        missing_metrics = set(KUBERNETES_METRIC_KEYS).difference(metric_responses)
        if missing_metrics:
            raise ValueError(
                f"Missing Kubernetes metrics in {path.name}: "
                f"{', '.join(sorted(missing_metrics))}"
            )

        for metric_key in KUBERNETES_METRIC_KEYS:
            results = metric_responses[metric_key].get("result")
            if not isinstance(results, list):
                raise ValueError(
                    f"Unexpected response for {metric_key} in {path.name}"
                )
            for result in results:
                for series in result.get("data", []):
                    dimensions = series.get("dimensions", [])
                    timestamps = series.get("timestamps", [])
                    values = series.get("values", [])
                    if len(dimensions) != 1 or len(timestamps) != len(values):
                        raise ValueError(
                            f"Unexpected series shape for {metric_key} in {path.name}"
                        )
                    (workload,) = map(str, dimensions)
                    for timestamp, value in zip(timestamps, values):
                        point_key = (
                            metric_key,
                            int(timestamp),
                            workload,
                        )
                        latest_points[point_key] = value

    metric_names = {metric_key: metric_key for metric_key in KUBERNETES_METRIC_KEYS}
    rows = []
    for (metric_key, timestamp, workload), value in latest_points.items():
        if value is None and metric_key in {
            "builtin:kubernetes.workload.cpu_usage",
            "builtin:kubernetes.workload.memory_working_set",
            "builtin:kubernetes.container.restarts",
            "builtin:kubernetes.container.oom_kills",
        }:
            value = 0
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, unit="ms", utc=True),
                "workload": workload,
                "metric": metric_names[metric_key],
                "value": pd.to_numeric(value, errors="coerce"),
            }
        )
    if not rows:
        raise ValueError("Dynatrace returned no Kubernetes metric series")

    frame = pd.DataFrame(rows)
    frame["value"] = frame["value"].astype(float)
    aggregate = frame.groupby(["timestamp", "workload", "metric"], as_index=False)[
        "value"
    ].sum(min_count=1)
    values = aggregate.pivot(
        index=["timestamp", "workload"],
        columns="metric",
        values="value",
    ).reset_index()
    values.columns.name = None

    required_gauges = (
        "builtin:kubernetes.workload.cpu_usage",
        "builtin:kubernetes.workload.requests_cpu",
        "builtin:kubernetes.workload.limits_cpu",
        "builtin:kubernetes.workload.memory_working_set",
        "builtin:kubernetes.workload.requests_memory",
        "builtin:kubernetes.workload.limits_memory",
    )
    for metric in required_gauges:
        if metric not in values:
            values[metric] = np.nan

    for metric, column in (
        ("builtin:kubernetes.container.restarts", "pod_restarts"),
        ("builtin:kubernetes.container.oom_kills", "oom_kills"),
    ):
        values[column] = values[metric] if metric in values else 0.0

    values["cpu_saturation_pct"] = (
        values["builtin:kubernetes.workload.cpu_usage"]
        / values["builtin:kubernetes.workload.requests_cpu"].where(
            values["builtin:kubernetes.workload.requests_cpu"] > 0
        )
        * 100
    )
    values["cpu_limit_pct"] = (
        values["builtin:kubernetes.workload.cpu_usage"]
        / values["builtin:kubernetes.workload.limits_cpu"].where(
            values["builtin:kubernetes.workload.limits_cpu"] > 0
        )
        * 100
    )
    values["memory_limit_pct"] = (
        values["builtin:kubernetes.workload.memory_working_set"]
        / values["builtin:kubernetes.workload.limits_memory"].where(
            values["builtin:kubernetes.workload.limits_memory"] > 0
        )
        * 100
    )
    values["memory_requests_pct"] = (
        values["builtin:kubernetes.workload.memory_working_set"]
        / values["builtin:kubernetes.workload.requests_memory"].where(
            values["builtin:kubernetes.workload.requests_memory"] > 0
        )
        * 100
    )
    values["service"] = values["workload"].map(
        {workload: service for service, workload in SERVICE_WORKLOADS.items()}
    )
    values = values.dropna(subset=["service"])
    if values.empty:
        raise ValueError(
            "Kubernetes workloads did not match the confirmed Astroshop service mapping"
        )
    return values.sort_values(["service", "timestamp"]).reset_index(drop=True)


def _service_features(service_frame: pd.DataFrame) -> pd.DataFrame:
    service = service_frame.set_index("timestamp").sort_index()
    grid = pd.date_range(
        start=service.index.min(),
        end=service.index.max(),
        freq=INTERVAL,
        tz="UTC",
    )
    service = service.reindex(grid)
    service.index.name = "timestamp"
    service["service"] = service_frame["service"].iloc[0]
    service["observed"] = service["total"].notna() & service["errors5xx"].notna()

    for column in (
        "cpu_saturation_pct",
        "cpu_limit_pct",
        "memory_limit_pct",
        "memory_requests_pct",
        "pod_restarts",
        "oom_kills",
    ):
        if column in service_frame:
            aligned = service_frame.set_index("timestamp")[column]
            service[column] = aligned.reindex(grid)
        else:
            service[column] = np.nan

    service["error_rate_5xx"] = service["errors5xx"] / service["total"].where(
        service["total"] > 0
    )
    service["request_rate"] = service["total"] / 300.0
    service["request_rate_mean_60m"] = service["request_rate"].rolling(
        WINDOW_60M, min_periods=WINDOW_60M
    ).mean()
    service["request_rate_delta_30m"] = (
        service["request_rate"] - service["request_rate"].shift(6)
    )
    service["error_rate_5xx_mean_60m"] = service["error_rate_5xx"].rolling(
        WINDOW_60M, min_periods=WINDOW_60M
    ).mean()
    service["error_rate_5xx_max_60m"] = service["error_rate_5xx"].rolling(
        WINDOW_60M, min_periods=WINDOW_60M
    ).max()
    service["error_rate_5xx_delta_30m"] = (
        service["error_rate_5xx"] - service["error_rate_5xx"].shift(6)
    )

    for metric in ("p95_latency_ms", "p99_latency_ms"):
        service[f"{metric}_mean_60m"] = service[metric].rolling(
            WINDOW_60M, min_periods=WINDOW_60M
        ).mean()
        service[f"{metric}_max_60m"] = service[metric].rolling(
            WINDOW_60M, min_periods=WINDOW_60M
        ).max()

    for metric, feature in (
        ("cpu_saturation_pct", "cpu_saturation"),
        ("cpu_limit_pct", "cpu_limit_pct"),
        ("memory_limit_pct", "memory_limit_pct"),
        ("memory_requests_pct", "memory_requests_pct"),
    ):
        service[f"{feature}_mean_60m"] = service[metric].rolling(
            WINDOW_60M, min_periods=WINDOW_60M
        ).mean()
        service[f"{feature}_max_60m"] = service[metric].rolling(
            WINDOW_60M, min_periods=WINDOW_60M
        ).max()

    service["pod_restarts"] = service["pod_restarts"].fillna(0)
    service["oom_kills"] = service["oom_kills"].fillna(0)
    service["pod_restarts_30m"] = service["pod_restarts"].rolling(
        6, min_periods=6
    ).sum()
    service["oom_kills_30m"] = service["oom_kills"].rolling(6, min_periods=6).sum()
    service["oom_any_30m"] = (service["oom_kills_30m"] > 0).astype(float)

    rolling_total = service["total"].rolling(WINDOW_60M, min_periods=WINDOW_60M).sum()
    rolling_errors = service["errors5xx"].rolling(WINDOW_60M, min_periods=WINDOW_60M).sum()
    rolling_observed = service["observed"].rolling(
        WINDOW_60M, min_periods=WINDOW_60M
    ).sum()
    complete_window = rolling_observed == WINDOW_60M
    service["slo_current"] = (1 - rolling_errors / rolling_total.where(rolling_total > 0)).where(
        complete_window
    )
    service["slo_error_budget_remaining_pct"] = (
        (SLO_TARGET - (rolling_errors / rolling_total.where(rolling_total > 0)))
        / (1 - SLO_TARGET)
        * 100
    ).where(complete_window)

    future_violation = pd.Series(False, index=service.index, dtype=bool)
    future_complete = pd.Series(True, index=service.index, dtype=bool)
    for offset in range(1, HORIZON_30M + 1):
        future_slo = service["slo_current"].shift(-offset)
        future_violation |= future_slo.notna() & (future_slo < SLO_TARGET)
        future_complete &= future_slo.notna()

    valid_features = service[FEATURE_COLUMNS].notna().all(axis=1)
    current_slo_healthy = service["slo_current"] >= SLO_TARGET
    label_valid = future_complete & current_slo_healthy & valid_features
    service["y"] = pd.Series(pd.NA, index=service.index, dtype="Int64")
    service.loc[label_valid, "y"] = future_violation.loc[label_valid].astype("int64")

    return service.reset_index()


def build_dataset(
    observations: pd.DataFrame,
    kubernetes_observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if observations.empty:
        raise ValueError("No observations provided")
    if kubernetes_observations is not None and not kubernetes_observations.empty:
        observations = observations.merge(
            kubernetes_observations[
                [
                    "timestamp",
                    "service",
                    "cpu_saturation_pct",
                    "cpu_limit_pct",
                    "memory_limit_pct",
                    "memory_requests_pct",
                    "pod_restarts",
                    "oom_kills",
                ]
            ],
            on=["timestamp", "service"],
            how="left",
            validate="one_to_one",
        )
    groups = [
        _service_features(group.copy())
        for _, group in observations.groupby("service", sort=True)
    ]
    dataset = pd.concat(groups, ignore_index=True)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS, how="all")
    return dataset.sort_values(["timestamp", "service"]).reset_index(drop=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw_path = Path(os.getenv("RAW_DATA_DIR", RAW_PATH))
    output_path = Path(os.getenv("DATASET_PATH", PROCESSED_PATH / "training_dataset.csv"))
    observations = load_request_observations(raw_path)
    dataset = build_dataset(observations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    trainable = int(dataset["y"].notna().sum())
    positives = int(dataset.loc[dataset["y"].notna(), "y"].sum())
    logging.info(
        "Wrote %d rows (%d trainable, %d positive labels) to %s",
        len(dataset),
        trainable,
        positives,
        output_path,
    )
    if trainable == 0:
        logging.warning(
            "No trainable labels yet; require a complete 60-minute window and future horizon"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
