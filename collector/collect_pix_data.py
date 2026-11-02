import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .dynatrace_client import DynatraceClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw"
LOOKBACK_PATTERN = re.compile(r"^[1-9][0-9]*[smhd]$")
EXPECTED_INTERVAL_MS = 5 * 60 * 1000
METRIC_SELECTORS = {
    "total": 'builtin:service.requestCount.total:splitBy("dt.entity.service")',
    "errors5xx": 'builtin:service.errors.fivexx.count:splitBy("dt.entity.service")',
    "p95_latency_ms": (
        'builtin:service.response.time:percentile(95):splitBy("dt.entity.service")'
    ),
    "p99_latency_ms": (
        'builtin:service.response.time:percentile(99):splitBy("dt.entity.service")'
    ),
}
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
KUBERNETES_SPLIT_DIMENSION = "k8s.workload.name"
SERVICE_WORKLOADS = {
    "frontend (frontend)": "frontend",
    "product-catalog (oteldemo.ProductCatalogService)": "product-catalog",
    "astroshop-currency": "currency",
    "astroshop-flagd": "flagd",
    "astroshop-cart": "cart",
    "astroshop-recommendation": "recommendation",
    "astroshop-image-provider": "image-provider",
    "ad (oteldemo.AdService)": "ad",
    "astroshop-shipping": "shipping",
    "astroshop-quote": "quote",
    "astroshop-fraud-detection": "fraud-detection",
    "astroshop-email": "email",
    "checkout (oteldemo.CheckoutService)": "checkout",
    "payment (oteldemo.PaymentService)": "payment",
    "accounting (Consumer`2)": "accounting",
}
SERVICE_NAME_ALIASES = {"astroshop-frontend": "frontend (frontend)"}
HTTP_SLI_SERVICE_NAMES = {
    "frontend (frontend)",
    "astroshop-cart",
    "astroshop-image-provider",
    "astroshop-shipping",
    "astroshop-quote",
    "astroshop-email",
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def validate_collection_settings(service_filter: str, lookback: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_. -]+", service_filter):
        raise ValueError(
            "DT_SERVICE_NAME_FILTER must contain only letters, numbers, spaces, "
            "dots, underscores, and hyphens"
        )
    if not LOOKBACK_PATTERN.fullmatch(lookback):
        raise ValueError("DT_LOOKBACK must be a positive duration such as 1h or 24h")


def validate_kubernetes_namespace(namespace: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", namespace):
        raise ValueError(
            "DT_KUBERNETES_NAMESPACE must contain only letters, numbers, "
            "dots, underscores, and hyphens"
        )


def kubernetes_metric_selectors(namespace: str) -> dict[str, str]:
    validate_kubernetes_namespace(namespace)
    return {
        metric_key: (
            f'{metric_key}:filter(eq("k8s.namespace.name","{namespace}"))'
            f':splitBy("{KUBERNETES_SPLIT_DIMENSION}")'
        )
        for metric_key in KUBERNETES_METRIC_KEYS
    }


def _metric_series_by_service(response: dict) -> dict[str, dict[int, float | None]]:
    results = response.get("result")
    if not isinstance(results, list):
        raise ValueError("Dynatrace metric response did not include a result list")

    series_by_service = {}
    for result in results:
        for data in result.get("data", []):
            dimensions = data.get("dimensions", [])
            timestamps = data.get("timestamps", [])
            values = data.get("values", [])
            if not dimensions or len(timestamps) != len(values):
                raise ValueError("Dynatrace returned an unexpected metric series shape")
            service_id = str(dimensions[0])
            series = series_by_service.setdefault(service_id, {})
            series.update(zip(timestamps, values))
    return series_by_service


def validate_five_minute_resolution(
    metric_responses: dict[str, dict],
    service_ids: set[str],
) -> None:
    observed_intervals = []
    for metric_name, response in metric_responses.items():
        results = response.get("result")
        if not isinstance(results, list):
            raise ValueError(
                f"Dynatrace metric response for {metric_name} did not include a result list"
            )
        for result in results:
            for data in result.get("data", []):
                dimensions = data.get("dimensions", [])
                if not dimensions or str(dimensions[0]) not in service_ids:
                    continue
                timestamps = data.get("timestamps", [])
                if len(timestamps) < 2:
                    continue
                try:
                    ordered = sorted(int(timestamp) for timestamp in timestamps)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Dynatrace returned invalid timestamps for {metric_name}"
                    ) from error
                intervals = [
                    later - earlier
                    for earlier, later in zip(ordered, ordered[1:])
                    if later > earlier
                ]
                if intervals:
                    observed_intervals.append((metric_name, intervals))

    coarse_series = []
    for metric_name, intervals in observed_intervals:
        median_interval = sorted(intervals)[len(intervals) // 2]
        if median_interval > EXPECTED_INTERVAL_MS:
            coarse_series.append((metric_name, median_interval))
    if coarse_series:
        metric_name, interval_ms = max(coarse_series, key=lambda item: item[1])
        interval_minutes = interval_ms / 60_000
        raise ValueError(
            "Dynatrace returned coarser-than-5-minute metric data "
            f"(median interval {interval_minutes:g} minutes for {metric_name}); "
            "collect a shorter lookback and verify timestamps before training"
        )


def build_observations(
    metric_responses: dict[str, dict],
    service_entities: list[dict],
    service_filter: str = "astroshop",
) -> list[dict]:
    eligible_names = {name.casefold() for name in HTTP_SLI_SERVICE_NAMES}
    matching_services = {
        str(entity["entityId"]): str(entity["displayName"])
        for entity in service_entities
        if entity.get("entityId")
        and entity.get("displayName")
        and (
            service_filter.casefold() in str(entity["displayName"]).casefold()
            or SERVICE_NAME_ALIASES.get(
                str(entity["displayName"]), str(entity["displayName"])
            ).casefold()
            == "frontend (frontend)"
        )
        and SERVICE_NAME_ALIASES.get(
            str(entity["displayName"]), str(entity["displayName"])
        ).casefold()
        in eligible_names
        and (entity.get("properties") or {}).get("serviceType") == "WEB_REQUEST_SERVICE"
    }
    if not matching_services:
        raise ValueError(
            f"No WEB_REQUEST_SERVICE entities matched {service_filter!r}; "
            "check DT_SERVICE_NAME_FILTER"
        )

    metrics = {
        name: _metric_series_by_service(response)
        for name, response in metric_responses.items()
    }
    required_metrics = set(METRIC_SELECTORS)
    missing_metrics = required_metrics.difference(metrics)
    if missing_metrics:
        raise ValueError(
            f"Missing metric responses: {', '.join(sorted(missing_metrics))}"
        )
    validate_five_minute_resolution(metric_responses, set(matching_services))

    observations = []
    for service_id, service_name in matching_services.items():
        service_series = {name: values.get(service_id) for name, values in metrics.items()}
        if any(series is None for series in service_series.values()):
            logging.info("%s skipped: one or more required metrics are unavailable", service_name)
            continue

        timestamps = set.intersection(*(set(series) for series in service_series.values()))
        for timestamp in sorted(timestamps):
            total = service_series["total"][timestamp]
            errors = service_series["errors5xx"][timestamp]
            p95 = service_series["p95_latency_ms"][timestamp]
            p99 = service_series["p99_latency_ms"][timestamp]
            if total is None or errors is None or p95 is None or p99 is None:
                continue
            observations.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        timestamp / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                    "service": SERVICE_NAME_ALIASES.get(service_name, service_name),
                    "total": total,
                    "errors5xx": errors,
                    "p95_latency_ms": p95 / 1000,
                    "p99_latency_ms": p99 / 1000,
                }
            )

    if not observations:
        raise ValueError(
            "No complete Astroshop request, HTTP 5xx, and latency metric bins were returned"
        )
    return observations


def save_result(
    service_filter: str,
    lookback: str,
    observations: list[dict],
    raw_path: Path = RAW_PATH,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_filter = re.sub(r"[^A-Za-z0-9_.-]+", "_", service_filter)
    output_path = raw_path / f"{safe_filter}_request_counts_5m_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "service_name_filter": service_filter,
        "lookback": lookback,
        "source": "Dynatrace Metrics API v2",
        "metric_selectors": METRIC_SELECTORS,
        "response": {"result": {"records": observations}},
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return output_path


def save_kubernetes_result(
    namespace: str,
    lookback: str,
    metric_selectors: dict[str, str],
    metric_responses: dict[str, dict],
    raw_path: Path = RAW_PATH,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = raw_path / f"kubernetes_container_metrics_5m_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "namespace": namespace,
        "lookback": lookback,
        "resolution": "5m",
        "source": "Dynatrace Metrics API v2",
        "metric_selectors": metric_selectors,
        "metric_responses": metric_responses,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env_file(PROJECT_ROOT / ".env")

    service_filter = os.getenv("DT_SERVICE_NAME_FILTER", "astroshop").strip()
    kubernetes_namespace = os.getenv("DT_KUBERNETES_NAMESPACE", "astroshop").strip()
    lookback = os.getenv("DT_LOOKBACK", "24h").strip()
    resolution = os.getenv("DT_METRIC_RESOLUTION", "5m").strip()
    raw_path = Path(os.getenv("RAW_DATA_DIR", RAW_PATH))
    validate_collection_settings(service_filter, lookback)
    validate_kubernetes_namespace(kubernetes_namespace)
    if resolution != "5m":
        raise ValueError("DT_METRIC_RESOLUTION is fixed at 5m for this prototype")

    client = DynatraceClient.from_environment()
    logging.info(
        "Collecting %s web request services from the last %s at %s resolution",
        service_filter,
        lookback,
        resolution,
    )

    service_entities = client.list_service_entities()
    metric_responses = {
        name: client.query_metric(selector, resolution, f"now-{lookback}")
        for name, selector in METRIC_SELECTORS.items()
    }
    observations = build_observations(metric_responses, service_entities, service_filter)
    output_path = save_result(service_filter, lookback, observations, raw_path)
    kubernetes_selectors = kubernetes_metric_selectors(kubernetes_namespace)
    kubernetes_responses = {
        metric_key: client.query_metric(selector, resolution, f"now-{lookback}")
        for metric_key, selector in kubernetes_selectors.items()
    }
    kubernetes_output_path = save_kubernetes_result(
        kubernetes_namespace,
        lookback,
        kubernetes_selectors,
        kubernetes_responses,
        raw_path,
    )
    service_count = len({row["service"] for row in observations})
    logging.info(
        "Saved %d metric bins for %d services to %s",
        len(observations),
        service_count,
        output_path,
    )
    logging.info(
        "Saved raw Kubernetes metric responses for namespace %s to %s",
        kubernetes_namespace,
        kubernetes_output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
