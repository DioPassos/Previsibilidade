import unittest
from unittest.mock import patch

from collector.collect_pix_data import METRIC_SELECTORS, validate_collection_settings
from collector.dynatrace_client import (
    DynatraceClient,
    authorization_scheme_for_token,
    platform_base_url,
    tenant_api_base_url,
)


class DynatraceConfigurationTests(unittest.TestCase):
    def test_classic_saas_url_maps_to_platform_host(self):
        self.assertEqual(
            platform_base_url("https://environment.live.dynatrace.com"),
            "https://environment.apps.dynatrace.com",
        )

    def test_platform_url_is_preserved(self):
        self.assertEqual(
            platform_base_url("https://environment.apps.dynatrace.com/"),
            "https://environment.apps.dynatrace.com",
        )

    def test_platform_url_maps_to_classic_api_tenant_host(self):
        self.assertEqual(
            tenant_api_base_url("https://environment.apps.dynatrace.com"),
            "https://environment.live.dynatrace.com",
        )

    def test_classic_api_tenant_url_is_preserved(self):
        self.assertEqual(
            tenant_api_base_url("https://environment.live.dynatrace.com"),
            "https://environment.live.dynatrace.com",
        )

    def test_platform_ui_path_and_query_are_removed(self):
        self.assertEqual(
            platform_base_url(
                "https://environment.apps.dynatrace.com/ui/apps/dynatrace/"
                "?gtf=-2h&gf=all"
            ),
            "https://environment.apps.dynatrace.com",
        )

    def test_unsupported_url_is_rejected(self):
        with self.assertRaises(ValueError):
            platform_base_url("https://example.com")

    def test_client_accepts_tenant_api_token_variable(self):
        with patch.dict(
            "os.environ",
            {
                "DT_TENANT_URL": "https://environment.live.dynatrace.com",
                "DT_TENANT_API_TOKEN": "dt0c01.test-token",
            },
            clear=True,
        ):
            client = DynatraceClient.from_environment()
        self.assertEqual(client.api_token, "dt0c01.test-token")

    def test_platform_token_uses_bearer_authorization(self):
        client = DynatraceClient(
            "https://environment.apps.dynatrace.com",
            "dt0s01.test-token",
        )
        self.assertEqual(
            client.session.headers["Authorization"],
            "Bearer dt0s01.test-token",
        )

    def test_classic_api_token_uses_api_token_authorization(self):
        client = DynatraceClient(
            "https://environment.apps.dynatrace.com",
            "dt0c01.test-token",
        )
        self.assertEqual(
            client.session.headers["Authorization"],
            "Api-Token dt0c01.test-token",
        )

    def test_unknown_token_type_is_rejected(self):
        with self.assertRaises(ValueError):
            authorization_scheme_for_token("unknown.token")

    def test_collector_uses_supported_http_5xx_and_service_metrics(self):
        self.assertEqual(
            set(METRIC_SELECTORS),
            {"total", "errors5xx", "p95_latency_ms", "p99_latency_ms"},
        )
        self.assertIn("builtin:service.requestCount.total", METRIC_SELECTORS["total"])
        self.assertIn("builtin:service.errors.fivexx.count", METRIC_SELECTORS["errors5xx"])
        self.assertIn("dt.entity.service", METRIC_SELECTORS["total"])

    def test_collection_settings_reject_invalid_filter_and_lookback(self):
        with self.assertRaises(ValueError):
            validate_collection_settings('astroshop" | fetch spans', "24h")
        with self.assertRaises(ValueError):
            validate_collection_settings("astroshop", "now() - 30d")


if __name__ == "__main__":
    unittest.main()
