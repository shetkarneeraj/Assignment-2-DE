import logging
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional

import requests

from .config import ApiConfig


logger = logging.getLogger(__name__)


class OpenElectricityClient:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.session = requests.Session()
        headers = {"Accept": "application/json"}
        if config.api_key:
            api_key = config.api_key.strip()
            if not api_key:
                logger.warning("API key is empty after stripping whitespace")
            else:
                prefix = config.api_key_prefix if config.api_key_prefix is not None else "Bearer"
                header_name = config.api_key_header or "Authorization"
                # Ensure proper format: "Bearer <token>" with exactly one space
                prefix = prefix.rstrip()  # Remove trailing space if present
                headers[header_name] = f"{prefix} {api_key}"
                logger.debug("Authorization header configured for API requests")
        else:
            logger.warning("No API key configured. API requests may fail with authentication errors.")
        self.session.headers.update(headers)

    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        logger.debug("Requesting %s with params %s", url, params)
        # Log auth status (but not the actual token for security)
        auth_header = self.session.headers.get("Authorization")
        if auth_header:
            # Show only first few chars for debugging
            masked = auth_header[:20] + "..." if len(auth_header) > 20 else auth_header
            logger.debug("Authorization header present: %s", masked)
        else:
            logger.warning("No Authorization header found in session headers")
        response = self.session.request(
            method, url, params=params, timeout=30
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "OpenElectricity API request failed: %s - %s", exc, response.text
            )
            raise
        return response.json()

    def fetch_facilities(self, network: Optional[str] = None) -> List[Dict]:
        params = {}
        if network or self.config.network:
            # API expects 'network_id' as the parameter name
            params["network_id"] = network or self.config.network
        payload = self._request("GET", self.config.facility_endpoint, params=params)
        facilities = payload.get("data", payload)
        logger.info("Fetched %d facilities for network_id %s", len(facilities), params.get("network_id"))
        return facilities

    def fetch_metric_timeseries(
        self,
        facility_id: str,
        metric: str,
        start: datetime,
        end: datetime,
        interval: str = "PT5M",
        page_size: Optional[int] = None,
    ) -> Iterator[Dict]:
        """
        Retrieves metric timeseries for a facility across pages.
        Yields individual records to keep memory usage low.
        """
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "page_size": page_size or self.config.max_page_size,
        }

        path = self.config.timeseries_endpoint.format(
            facility_id=facility_id, metric=metric
        )

        next_path: Optional[str] = path
        try:
            while next_path:
                payload = self._request("GET", next_path, params=params)
                for item in payload.get("data", []):
                    yield item
                next_path = payload.get("next")
                params = None  # Subsequent requests rely on next link
        except requests.HTTPError as exc:
            # Log 404 errors but don't crash - metric may not be available for this facility
            if exc.response.status_code == 404:
                logger.debug(
                    "Metric %s not available for facility %s (404 Not Found)",
                    metric,
                    facility_id,
                )
                return
            raise

    def fetch_metrics_for_facilities(
        self,
        facilities: Iterable[Dict],
        metrics: Iterable[tuple[str, str]],
        start: datetime,
        end: datetime,
        interval: str = "PT5M",
    ) -> Iterator[Dict]:
        for facility in facilities:
            # OpenElectricity API uses 'code' as the facility identifier
            facility_id = facility.get("id") or facility.get("facility_id") or facility.get("code")
            if not facility_id:
                logger.warning("Skipping facility without id/code: %s", facility)
                continue
            for logical_metric, remote_metric in metrics:
                for record in self.fetch_metric_timeseries(
                    facility_id=facility_id,
                    metric=remote_metric,
                    start=start,
                    end=end,
                    interval=interval,
                ):
                    record.setdefault("facility_id", facility_id)
                    record["metric"] = logical_metric
                    yield record
