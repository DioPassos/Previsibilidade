import os
import time
from urllib.parse import urlsplit, urlunsplit

import requests


class DynatraceApiError(RuntimeError):
    pass


def authorization_scheme_for_token(token: str) -> str:
    prefix = token.strip().split(".", 1)[0]
    if prefix == "dt0s01":
        return "Bearer"
    if prefix == "dt0c01":
        return "Api-Token"
    raise ValueError("Unsupported Dynatrace token type; expected a dt0s01 or dt0c01 token")


def platform_base_url(tenant_url: str) -> str:
    parsed = urlsplit(tenant_url.strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("DT_TENANT_URL must be an absolute HTTP(S) Dynatrace URL")
    if parsed.username or parsed.password:
        raise ValueError("DT_TENANT_URL must not contain embedded credentials")

    hostname = parsed.hostname.lower()
    if hostname.endswith(".live.dynatrace.com"):
        hostname = hostname.removesuffix(".live.dynatrace.com") + ".apps.dynatrace.com"
    elif not (hostname.endswith(".apps.dynatrace.com") or hostname == "apps.dynatrace.com"):
        raise ValueError(
            "DT_TENANT_URL must be a Dynatrace SaaS .live.dynatrace.com or "
            ".apps.dynatrace.com URL"
        )

    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def tenant_api_base_url(tenant_url: str) -> str:
    parsed = urlsplit(tenant_url.strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("DT_TENANT_URL must be an absolute HTTP(S) Dynatrace URL")
    if parsed.username or parsed.password:
        raise ValueError("DT_TENANT_URL must not contain embedded credentials")

    hostname = parsed.hostname.lower()
    if hostname.endswith(".apps.dynatrace.com"):
        hostname = hostname.removesuffix(".apps.dynatrace.com") + ".live.dynatrace.com"
    elif not (hostname.endswith(".live.dynatrace.com") or hostname == "live.dynatrace.com"):
        raise ValueError(
            "DT_TENANT_URL must be a Dynatrace SaaS .live.dynatrace.com or "
            ".apps.dynatrace.com URL"
        )

    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


class DynatraceClient:
    def __init__(self, tenant_url: str, api_token: str, timeout: int = 30):
        self.base_url = platform_base_url(tenant_url)
        self.api_base_url = tenant_api_base_url(tenant_url)
        self.api_token = api_token.strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": (
                    f"{authorization_scheme_for_token(self.api_token)} {self.api_token}"
                ),
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    @classmethod
    def from_environment(cls):
        tenant_url = os.getenv("DT_TENANT_URL", "").strip()
        api_token = (
            os.getenv("DT_API_TOKEN") or os.getenv("DT_TENANT_API_TOKEN", "")
        ).strip()
        if not tenant_url:
            raise ValueError("DT_TENANT_URL is required")
        if not api_token:
            raise ValueError("DT_API_TOKEN or DT_TENANT_API_TOKEN is required")
        return cls(tenant_url, api_token)

    def execute_query(self, query: str) -> dict:
        endpoint = f"{self.base_url}/platform/storage/query/v1/query:execute"
        try:
            response = self.session.post(
                endpoint,
                json={"query": query},
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise DynatraceApiError("Could not connect to the Dynatrace query API") from error
        payload = self._decode_response(response)

        if not isinstance(payload, dict):
            raise DynatraceApiError("Dynatrace returned an unexpected response format")
        if payload.get("error"):
            raise DynatraceApiError(f"Dynatrace query failed: {payload['error']}")

        request_token = payload.get("requestToken") or payload.get("request-token")
        state = str(payload.get("state", "")).upper()
        if request_token and state not in {"SUCCEEDED", "SUCCESS"}:
            payload = self._poll_query(request_token)

        if not payload.get("result") and "records" not in payload:
            raise DynatraceApiError("Dynatrace response did not include query results")
        return payload

    def query_metric(self, selector: str, resolution: str, from_time: str) -> dict:
        return self._get_api_json(
            "/api/v2/metrics/query",
            params={
                "metricSelector": selector,
                "resolution": resolution,
                "from": from_time,
                "to": "now",
            },
        )

    def list_service_entities(self) -> list[dict]:
        endpoint = f"{self.api_base_url}/api/v2/entities"
        params = {
            "entitySelector": 'type("SERVICE")',
            "fields": "+properties",
            "pageSize": "500",
        }
        entities = []
        while True:
            payload = self._get_api_json(endpoint, params=params)
            page = payload.get("entities")
            if not isinstance(page, list):
                raise DynatraceApiError("Dynatrace entity response did not include entities")
            entities.extend(page)
            next_page_key = payload.get("nextPageKey")
            if not next_page_key:
                return entities
            params = {"nextPageKey": next_page_key}

    def _get_api_json(self, path: str, params: dict[str, str]) -> dict:
        endpoint = path if path.startswith("https://") else f"{self.api_base_url}{path}"
        try:
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
        except requests.RequestException as error:
            raise DynatraceApiError("Could not connect to the Dynatrace API") from error
        payload = self._decode_response(response)
        if not isinstance(payload, dict):
            raise DynatraceApiError("Dynatrace returned an unexpected response format")
        if payload.get("error"):
            raise DynatraceApiError(f"Dynatrace API failed: {payload['error']}")
        return payload

    def _poll_query(self, request_token: str) -> dict:
        endpoint = f"{self.base_url}/platform/storage/query/v1/query:poll"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                response = self.session.get(
                    endpoint,
                    params={"request-token": request_token},
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                raise DynatraceApiError("Could not poll the Dynatrace query API") from error
            payload = self._decode_response(response)
            if not isinstance(payload, dict):
                raise DynatraceApiError("Dynatrace returned an unexpected polling response")
            state = str(payload.get("state", "")).upper()
            if payload.get("error") or state in {"FAILED", "CANCELLED"}:
                raise DynatraceApiError(f"Dynatrace query failed: {payload.get('error', state)}")
            if state in {"SUCCEEDED", "SUCCESS"} or payload.get("result"):
                return payload
            time.sleep(1)
        raise DynatraceApiError("Timed out waiting for Dynatrace query results")

    def _decode_response(self, response: requests.Response) -> dict:
        if not response.ok:
            detail = response.text[:1000].replace(self.api_token, "[redacted]")
            raise DynatraceApiError(
                f"Dynatrace API returned HTTP {response.status_code}: {detail}"
            )
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise DynatraceApiError("Dynatrace returned a non-JSON response") from error
