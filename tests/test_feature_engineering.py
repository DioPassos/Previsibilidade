import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from collector.collect_pix_data import KUBERNETES_METRIC_KEYS
from feature_engineering.build_dataset import (
    WINDOW_60M,
    build_dataset,
    load_kubernetes_observations,
)


class DatasetConstructionTests(unittest.TestCase):
    def make_observations(self, count: int = WINDOW_60M + 6 + 1) -> pd.DataFrame:
        timestamps = pd.date_range(
            "2025-01-01T00:00:00Z",
            periods=count,
            freq="5min",
        )
        errors = np.zeros(count, dtype=np.int64)
        errors[WINDOW_60M] = 100_000_000
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "service": "test-service",
                "total": 10_000_000,
                "errors5xx": errors,
                "p95_latency_ms": 40.0,
                "p99_latency_ms": 80.0,
            }
        )

    def test_loads_and_aggregates_kubernetes_series_by_workload(self):
        metric_values = {
            "builtin:kubernetes.workload.cpu_usage": 50.0,
            "builtin:kubernetes.workload.requests_cpu": 100.0,
            "builtin:kubernetes.workload.limits_cpu": 200.0,
            "builtin:kubernetes.workload.memory_working_set": 100.0,
            "builtin:kubernetes.workload.requests_memory": 150.0,
            "builtin:kubernetes.workload.limits_memory": 200.0,
            "builtin:kubernetes.container.restarts": 1.0,
            "builtin:kubernetes.container.oom_kills": 1.0,
        }
        metric_responses = {}
        for metric_key in KUBERNETES_METRIC_KEYS:
            metric_responses[metric_key] = {
                "result": [
                    {
                        "data": [
                            {
                                "dimensions": [
                                    "frontend",
                                ],
                                "timestamps": [1_760_000_000_000],
                                "values": [value],
                            }
                            for value in [metric_values[metric_key]]
                        ]
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kubernetes_container_metrics_5m_test.json"
            path.write_text(
                json.dumps(
                    {
                        "namespace": "astroshop",
                        "metric_responses": metric_responses,
                    }
                )
            )
            observations = load_kubernetes_observations(Path(directory))

        self.assertEqual(observations.loc[0, "service"], "frontend (frontend)")
        self.assertEqual(observations.loc[0, "cpu_saturation_pct"], 50.0)
        self.assertEqual(observations.loc[0, "cpu_limit_pct"], 25.0)
        self.assertEqual(observations.loc[0, "memory_limit_pct"], 50.0)
        self.assertEqual(observations.loc[0, "pod_restarts"], 1.0)
        self.assertEqual(observations.loc[0, "oom_kills"], 1.0)

    def test_one_hour_proxy_slo_crossing_is_labeled_within_next_hour(self):
        observations = self.make_observations()
        dataset = build_dataset(observations)
        before_crossing = dataset.loc[
            dataset["timestamp"] == observations.loc[WINDOW_60M - 1, "timestamp"]
        ].iloc[0]
        at_crossing = dataset.loc[
            dataset["timestamp"] == observations.loc[WINDOW_60M, "timestamp"]
        ].iloc[0]

        self.assertEqual(int(before_crossing["y"]), 1)
        self.assertGreaterEqual(before_crossing["slo_current"], 0.9995)
        self.assertLess(at_crossing["slo_current"], 0.9995)
        self.assertTrue(pd.isna(at_crossing["y"]))

    def test_missing_time_bin_is_not_treated_as_success(self):
        observations = self.make_observations()
        gap_timestamp = observations.loc[5, "timestamp"]
        observations = observations[observations["timestamp"] != gap_timestamp]
        dataset = build_dataset(observations)

        affected = dataset.loc[
            dataset["timestamp"] == observations.loc[WINDOW_60M, "timestamp"]
        ].iloc[0]
        self.assertTrue(pd.isna(affected["slo_current"]))
        self.assertTrue(pd.isna(affected["y"]))


if __name__ == "__main__":
    unittest.main()
