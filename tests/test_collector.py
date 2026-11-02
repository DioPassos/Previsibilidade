import json
import tempfile
import unittest
from pathlib import Path

from collector.collect_pix_data import (
    KUBERNETES_METRIC_KEYS,
    METRIC_SELECTORS,
    build_observations,
    kubernetes_metric_selectors,
    save_kubernetes_result,
)


class MetricCollectionTests(unittest.TestCase):
    def metric_response(self, value, service_id="SERVICE-1"):
        return {
            "result": [
                {
                    "data": [
                        {
                            "dimensions": [service_id],
                            "timestamps": [1_760_000_000_000],
                            "values": [value],
                        }
                    ]
                }
            ]
        }

    def test_normalizes_api_v2_series_for_astroshop_web_services(self):
        responses = {
            "total": self.metric_response(100),
            "errors5xx": self.metric_response(1),
            "p95_latency_ms": self.metric_response(12_000),
            "p99_latency_ms": self.metric_response(25_000),
        }
        entities = [
            {
                "entityId": "SERVICE-1",
                "displayName": "astroshop-frontend",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            },
            {
                "entityId": "SERVICE-2",
                "displayName": "astroshop-cart",
                "properties": {"serviceType": "UNIFIED"},
            },
        ]

        records = build_observations(responses, entities)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["service"], "frontend (frontend)")
        self.assertEqual(records[0]["total"], 100)
        self.assertEqual(records[0]["errors5xx"], 1)
        self.assertEqual(records[0]["p95_latency_ms"], 12)
        self.assertEqual(records[0]["p99_latency_ms"], 25)

    def test_includes_confirmed_service_names_without_astroshop_prefix(self):
        responses = {
            name: {
                "result": [
                    {
                        "data": [
                            {
                                "dimensions": ["SERVICE-FRONTEND"],
                                "timestamps": [1_760_000_000_000],
                                "values": [1],
                            },
                            {
                                "dimensions": ["SERVICE-AD"],
                                "timestamps": [1_760_000_000_000],
                                "values": [1],
                            },
                        ],
                    }
                ]
            }
            for name in METRIC_SELECTORS
        }
        entities = [
            {
                "entityId": "SERVICE-FRONTEND",
                "displayName": "frontend (frontend)",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            },
            {
                "entityId": "SERVICE-AD",
                "displayName": "ad (oteldemo.AdService)",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            },
        ]

        records = build_observations(responses, entities)

        self.assertEqual(
            {record["service"] for record in records},
            {"frontend (frontend)"},
        )

    def test_excludes_services_without_confirmed_http_sli_coverage(self):
        responses = {name: self.metric_response(0) for name in METRIC_SELECTORS}
        entities = [
            {
                "entityId": "SERVICE-FLAGD",
                "displayName": "astroshop-flagd",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            },
            {
                "entityId": "SERVICE-GRPC",
                "displayName": "product-catalog (oteldemo.ProductCatalogService)",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            },
        ]

        with self.assertRaisesRegex(ValueError, "No WEB_REQUEST_SERVICE"):
            build_observations(responses, entities)

    def test_rejects_downsampled_metric_series(self):
        timestamps = [1_760_000_000_000, 1_760_000_600_000, 1_760_001_200_000]
        responses = {}
        for name in METRIC_SELECTORS:
            response = self.metric_response(1)
            response["result"][0]["data"][0]["timestamps"] = timestamps
            response["result"][0]["data"][0]["values"] = [1, 1, 1]
            responses[name] = response
        entities = [
            {
                "entityId": "SERVICE-1",
                "displayName": "astroshop-cart",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            }
        ]

        with self.assertRaisesRegex(ValueError, "coarser-than-5-minute"):
            build_observations(responses, entities)

    def test_ignores_downsampled_series_for_ineligible_services(self):
        responses = {}
        timestamps = [1_760_000_000_000, 1_760_000_600_000, 1_760_001_200_000]
        for name in METRIC_SELECTORS:
            response = self.metric_response(1)
            response["result"][0]["data"].append(
                {
                    "dimensions": ["SERVICE-GRPC"],
                    "timestamps": timestamps,
                    "values": [1, 1, 1],
                }
            )
            responses[name] = response
        entities = [
            {
                "entityId": "SERVICE-1",
                "displayName": "astroshop-cart",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            },
            {
                "entityId": "SERVICE-GRPC",
                "displayName": "product-catalog (oteldemo.ProductCatalogService)",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            },
        ]

        records = build_observations(responses, entities)

        self.assertEqual({record["service"] for record in records}, {"astroshop-cart"})

    def test_rejects_empty_matching_services(self):
        with self.assertRaisesRegex(ValueError, "No WEB_REQUEST_SERVICE"):
            build_observations(
                {name: self.metric_response(1) for name in METRIC_SELECTORS},
                [],
            )

    def test_ignores_services_without_all_required_metrics(self):
        responses = {name: self.metric_response(1) for name in METRIC_SELECTORS}
        responses.pop("errors5xx")
        entities = [
            {
                "entityId": "SERVICE-1",
                "displayName": "astroshop-frontend",
                "properties": {"serviceType": "WEB_REQUEST_SERVICE"},
            }
        ]

        with self.assertRaisesRegex(ValueError, "Missing metric responses"):
            build_observations(responses, entities)

    def test_builds_kubernetes_metric_selectors_for_namespace(self):
        selectors = kubernetes_metric_selectors("astroshop")

        self.assertEqual(set(selectors), set(KUBERNETES_METRIC_KEYS))
        self.assertIn('eq("k8s.namespace.name","astroshop")', selectors[KUBERNETES_METRIC_KEYS[0]])
        self.assertIn(
            'splitBy("k8s.workload.name")',
            selectors[KUBERNETES_METRIC_KEYS[0]],
        )
        self.assertTrue(KUBERNETES_METRIC_KEYS[0].startswith("builtin:kubernetes.workload."))
        self.assertTrue(KUBERNETES_METRIC_KEYS[-1].endswith("oom_kills"))

    def test_rejects_unsafe_kubernetes_namespace(self):
        with self.assertRaisesRegex(ValueError, "DT_KUBERNETES_NAMESPACE"):
            kubernetes_metric_selectors('astroshop")')

    def test_saves_unmodified_kubernetes_metric_responses(self):
        selectors = kubernetes_metric_selectors("astroshop")
        responses = {key: {"result": [{"data": []}]} for key in selectors}

        with tempfile.TemporaryDirectory() as directory:
            output_path = save_kubernetes_result(
                "astroshop",
                "24h",
                selectors,
                responses,
                Path(directory),
            )
            payload = json.loads(output_path.read_text())

        self.assertEqual(payload["namespace"], "astroshop")
        self.assertEqual(payload["resolution"], "5m")
        self.assertEqual(payload["metric_responses"], responses)


if __name__ == "__main__":
    unittest.main()
