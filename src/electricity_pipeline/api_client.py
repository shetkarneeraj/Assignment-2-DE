import logging
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional

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
        response = self.session.request(
            method, url, params=params, timeout=30
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            # Handle 404 errors gracefully for facility data
            if exc.response.status_code == 404:
                logger.debug(
                    "API request returned 404 (resource not found): %s", url
                )
            else:
                logger.error(
                    "OpenElectricity API request failed: %s - %s", exc, response.text
                )
            raise
        return response.json()
    
    def fetch_facilities(self, network: Optional[str] = None) -> List[Dict]:
        """
        Fetch all facilities for a network.
        
        Endpoint: GET /v4/facilities/?network_id={network_code}
        Note: Trailing slash is required for this endpoint
        """
        params = {}
        if network or self.config.network:
            params["network_id"] = network or self.config.network
        
        # Facilities endpoint requires trailing slash
        payload = self._request("GET", "/facilities/", params=params)
        facilities = payload.get("data", payload)
        logger.info("Fetched %d facilities for network_id %s", len(facilities), params.get("network_id"))
        return facilities
    
    def fetch_facility_metrics(
        self,
        facility_code: str,
        metrics: List[str],
        start: datetime,
        end: datetime,
        interval: str = "5m",
    ) -> Iterator[Dict]:
        """
        Fetch metrics for a specific facility.
        
        Uses endpoint: GET /v4/data/facilities/{network_code}?facility_code={code}&metrics=...
        or: GET /v4/facilities/{facility_code}/metrics/{metric}
        
        Args:
            facility_code: Facility code identifier
            metrics: List of metrics to retrieve
            start: Start datetime (timezone-naive)
            end: End datetime (timezone-naive)
            interval: Time interval (5m, 1h, 1d, etc.)
        
        Yields:
            Dict records with facility_id, metric, timestamp, value
        """
        # Convert interval format
        interval_param = interval
        if interval.startswith("PT"):
            if interval == "PT5M":
                interval_param = "5m"
            elif interval == "PT1H":
                interval_param = "1h"
            elif interval == "PT1D":
                interval_param = "1d"
            else:
                interval_param = interval.replace("PT", "").replace("M", "m").replace("H", "h").replace("D", "d")
        
        # Remove timezone if present
        start_naive = start.replace(tzinfo=None) if start.tzinfo else start
        end_naive = end.replace(tzinfo=None) if end.tzinfo else end
        
        # API data limits
        max_days_per_request = {
            "5m": 7,
            "1h": 30,
            "1d": 365,
        }.get(interval_param, 365)
        
        max_range = timedelta(days=max_days_per_request)
        date_range = end_naive - start_naive
        
        # Split into multiple requests if exceeding limit
        if date_range > max_range:
            current_start = start_naive
            while current_start < end_naive:
                current_end = min(current_start + max_range, end_naive)
                yield from self._fetch_facility_metrics_single(
                    facility_code, metrics, current_start, current_end, interval_param
                )
                current_start = current_end
        else:
            yield from self._fetch_facility_metrics_single(
                facility_code, metrics, start_naive, end_naive, interval_param
            )
    
    def _fetch_facility_metrics_single(
        self,
        facility_code: str,
        metrics: List[str],
        start: datetime,
        end: datetime,
        interval_param: str,
    ) -> Iterator[Dict]:
        """
        Fetch metrics for a facility using individual metric endpoints.
        Format: /facilities/{facility_code}/metrics/{metric}?start=...&end=...&interval=...
        """
        for metric in metrics:
            try:
                params = {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "interval": interval_param,
                    "page_size": 288,  # Max page size
                }
                
                path = f"/facilities/{facility_code}/metrics/{metric}"
                
                try:
                    payload = self._request("GET", path, params=params)
                    data = payload.get("data", [])
                    
                    # Handle pagination - next might be a full URL or relative path
                    next_path = payload.get("next")
                    while next_path:
                        # If next is a full URL, extract just the path
                        if next_path.startswith("http"):
                            from urllib.parse import urlparse
                            parsed = urlparse(next_path)
                            next_path = parsed.path
                            # Remove base_url prefix if present
                            if next_path.startswith("/v4/"):
                                next_path = next_path[4:]
                        
                        next_payload = self._request("GET", next_path.lstrip("/"), params=None)
                        data.extend(next_payload.get("data", []))
                        next_path = next_payload.get("next")
                    
                    # Yield each record with facility_id
                    for item in data:
                        if isinstance(item, dict):
                            record = item.copy()
                            record["facility_id"] = facility_code
                            record["metric"] = metric
                            yield record
                            
                except requests.HTTPError as exc:
                    if exc.response.status_code == 404:
                        logger.info(
                            "Metric '%s' not available for facility '%s' (404)",
                            metric, facility_code
                        )
                        continue
                    raise
                    
            except Exception as e:
                logger.warning(
                    "Error fetching metric '%s' for facility '%s': %s",
                    metric, facility_code, e
                )
                continue
    
    def fetch_facilities_metrics(
        self,
        facilities: List[Dict],
        metrics: List[str],
        start: datetime,
        end: datetime,
        interval: str = "5m",
        network_code: Optional[str] = None,
    ) -> Iterator[Dict]:
        """
        Fetch metrics for multiple facilities using bulk endpoint.
        
        Uses: GET /v4/data/facilities/{network_code}?facility_code=...&metrics=...
        This is more efficient than individual facility requests.
        
        Args:
            facilities: List of facility dicts (from fetch_facilities)
            metrics: List of metric names
            start: Start datetime
            end: End datetime
            interval: Time interval
            network_code: Network code (defaults to config network)
        
        Yields:
            Records with facility_id, metric, timestamp, value
        """
        network = network_code or self.config.network
        if not network:
            raise ValueError("network_code must be specified or configured")
        
        # Convert interval format
        interval_param = interval
        if interval.startswith("PT"):
            if interval == "PT5M":
                interval_param = "5m"
            elif interval == "PT1H":
                interval_param = "1h"
            elif interval == "PT1D":
                interval_param = "1d"
            else:
                interval_param = interval.replace("PT", "").replace("M", "m").replace("H", "h").replace("D", "d")
        
        # Remove timezone if present
        start_naive = start.replace(tzinfo=None) if start.tzinfo else start
        end_naive = end.replace(tzinfo=None) if end.tzinfo else end
        
        # Get facility codes
        facility_codes = [f.get("code") for f in facilities if f.get("code")]
        logger.info(
            "Fetching metrics %s for %d facilities using bulk endpoint",
            metrics, len(facility_codes)
        )
        
        # API data limits
        max_days_per_request = {
            "5m": 7,
            "1h": 30,
            "1d": 365,
        }.get(interval_param, 365)
        
        max_range = timedelta(days=max_days_per_request)
        date_range = end_naive - start_naive
        
        # Split into date windows if needed
        date_windows = []
        if date_range > max_range:
            current_start = start_naive
            while current_start < end_naive:
                current_end = min(current_start + max_range, end_naive)
                date_windows.append((current_start, current_end))
                current_start = current_end
        else:
            date_windows = [(start_naive, end_naive)]
        
        # Filter out network-level metrics (price, demand) - they're not available at facility level
        # Facility-level metrics: power, emissions
        # Network-level metrics: price, demand
        facility_metrics = [m for m in metrics if m not in ["price", "demand"]]
        
        if not facility_metrics:
            logger.warning(
                "No facility-level metrics requested. Price and demand are network-level only."
            )
            return
        
        if len(facility_metrics) < len(metrics):
            logger.info(
                "Filtered out network-level metrics. Facility endpoint will fetch: %s",
                facility_metrics
            )
        
        # Process facilities in batches - use smaller batches and one metric at a time
        # to avoid 500 errors
        batch_size = 5  # Smaller batches
        
        # Process one metric at a time to avoid API overload
        total_batches = len(facility_metrics) * len(date_windows) * ((len(facility_codes) + batch_size - 1) // batch_size)
        batch_count = [0]  # Use list to allow modification in nested function
        
        for metric in facility_metrics:
            logger.info("Processing metric '%s' for %d facilities", metric, len(facility_codes))
            
            for date_start, date_end in date_windows:
                for i in range(0, len(facility_codes), batch_size):
                    batch = facility_codes[i:i + batch_size]
                    batch_count[0] += 1
                    if batch_count[0] % 10 == 0 or batch_count[0] == 1:
                        logger.info(
                            "Progress: Batch %d/%d - Processing facilities %d-%d for metric '%s'",
                            batch_count[0], total_batches,
                            i + 1, min(i + batch_size, len(facility_codes)), metric
                        )
                    else:
                        logger.debug(
                            "Fetching batch %d-%d of %d facilities for metric '%s'",
                            i + 1, min(i + batch_size, len(facility_codes)),
                            len(facility_codes), metric
                        )
                    
                    params = {
                        "facility_code": batch,  # Array of facility codes
                        "metrics": [metric],  # One metric at a time
                        "interval": interval_param,
                        "date_start": date_start.isoformat(),
                        "date_end": date_end.isoformat(),
                    }
                    
                    path = f"/data/facilities/{network}"
                    
                    try:
                        payload = self._request("GET", path, params=params)
                        data = payload.get("data", [])
                        
                        # Process response data - similar structure to network endpoint
                        # Each item in data[] is a metric object with results[]
                        for metric_obj in data:
                            if not isinstance(metric_obj, dict):
                                continue
                            
                            metric_name = metric_obj.get("metric")
                            results = metric_obj.get("results", [])
                            
                            # Process each result (each facility's data)
                            for result in results:
                                if not isinstance(result, dict):
                                    continue
                                
                                # Get facility code from columns
                                columns = result.get("columns", {})
                                facility_code = columns.get("unit_code") or columns.get("facility_code")
                                
                                if not facility_code:
                                    continue
                                
                                # Get timeseries data points
                                timeseries_data = result.get("data", [])
                                
                                # Convert [timestamp, value] arrays to records
                                for point in timeseries_data:
                                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                                        timestamp = point[0]
                                        value = point[1]
                                        
                                        record = {
                                            "facility_id": facility_code,
                                            "metric": metric_name,
                                            "timestamp": timestamp,
                                            "value": value,
                                        }
                                        
                                        yield record
                                    elif isinstance(point, dict):
                                        point_record = point.copy()
                                        point_record["facility_id"] = facility_code
                                        if metric_name and "metric" not in point_record:
                                            point_record["metric"] = metric_name
                                        yield point_record
                        
                    except requests.HTTPError as exc:
                        if exc.response.status_code in (404, 416):
                            # 404: Not Found, 416: Range Not Satisfiable (no data in time range)
                            # These are expected - many facilities don't have data
                            logger.debug(
                                "No data available for batch of %d facilities (status %d) - continuing",
                                len(batch), exc.response.status_code
                            )
                            continue
                        elif exc.response.status_code == 500:
                            logger.warning(
                                "Server error (500) for batch of %d facilities. Skipping batch.",
                                len(batch)
                            )
                            continue
                        # Log other errors at error level
                        logger.error(
                            "API error %d for batch: %s",
                            exc.response.status_code, exc.response.text[:200] if exc.response else str(exc)
                        )
                        raise

    def fetch_network_data(
        self,
        metrics: List[str],
        start: datetime,
        end: datetime,
        interval: str = "5m",
        network_code: Optional[str] = None,
        network_region: Optional[str] = None,
        primary_grouping: Optional[str] = None,
    ) -> Iterator[Dict]:
        """
        Retrieves market/network-level data for specified metrics.
        
        Endpoint: GET /v4/market/network/{network_code}
        See: https://docs.openelectricity.org.au/api-reference/market/get-network-data
        
        Args:
            metrics: List of metrics to retrieve (e.g., ["power", "emissions", "price", "demand"])
            start: Start datetime
            end: End datetime
            interval: Time interval (5m, 1h, 1d, etc.)
            network_code: Network code (defaults to config network)
            network_region: Optional network region filter
            primary_grouping: Optional grouping (network, network_region)
        
        Yields:
            Dict records with timeseries data
        """
        # Use configured network if not specified
        network = network_code or self.config.network
        if not network:
            raise ValueError("network_code must be specified or configured in ApiConfig")
        
        # Convert ISO 8601 duration to API format if needed
        interval_param = interval
        if interval.startswith("PT"):
            if interval == "PT5M":
                interval_param = "5m"
            elif interval == "PT1H":
                interval_param = "1h"
            elif interval == "PT1D":
                interval_param = "1d"
            else:
                interval_param = interval.replace("PT", "").replace("M", "m").replace("H", "h").replace("D", "d")
        
        # API data limits: 5-minute interval has max 7 days per request
        max_days_per_request = {
            "5m": 7,
            "1h": 30,
            "1d": 365,
        }.get(interval_param, 365)
        
        max_range = timedelta(days=max_days_per_request)
        date_range = end - start
        
        # Split into multiple requests if exceeding the limit
        if date_range > max_range:
            logger.debug(
                "Date range %s exceeds %d day limit for interval %s. Splitting network data request.",
                date_range, max_days_per_request, interval_param
            )
            current_start = start
            while current_start < end:
                current_end = min(current_start + max_range, end)
                yield from self._fetch_network_data_single_request(
                    network, metrics, current_start, current_end,
                    interval_param, network_region, primary_grouping
                )
                current_start = current_end
        else:
            yield from self._fetch_network_data_single_request(
                network, metrics, start, end,
                interval_param, network_region, primary_grouping
            )
    
    def _fetch_network_data_single_request(
        self,
        network_code: str,
        metrics: List[str],
        start: datetime,
        end: datetime,
        interval_param: str,
        network_region: Optional[str] = None,
        primary_grouping: Optional[str] = None,
    ) -> Iterator[Dict]:
        """
        Internal method to fetch network data for a single request (within API limits).
        
        Format: GET /v4/market/network/{network_code}?metrics=power&metrics=emissions&interval=5m&date_start=...&date_end=...&primary_grouping=network_region
        
        Note: The API expects metrics as an array (multiple metrics= parameters),
        not as a comma-separated string.
        """
        # Construct query parameters exactly as per API documentation
        # Metrics should be passed as an array (multiple metrics= parameters)
        # Format: metrics=power&metrics=emissions (not metrics=power,emissions)
        # The API expects metrics as an enum array, not a comma-separated string
        # Dates must be timezone-naive in network time (no Z or timezone info)
        # Error: "Date start must be timezone naive and in network time"
        if start.tzinfo is not None:
            # Remove timezone info - convert to naive datetime
            start = start.replace(tzinfo=None)
        if end.tzinfo is not None:
            # Remove timezone info - convert to naive datetime
            end = end.replace(tzinfo=None)
        
        start_str = start.isoformat()
        end_str = end.isoformat()
        
        params = {
            "metrics": metrics,  # Array format - requests will format as metrics=power&metrics=emissions
            "interval": interval_param,
            "date_start": start_str,
            "date_end": end_str,
        }
        
        # Optional parameters
        if network_region:
            params["network_region"] = network_region
        if primary_grouping:
            params["primary_grouping"] = primary_grouping
        
        # Endpoint: /market/network/{network_code}
        path = f"/market/network/{network_code}"
        
        try:
            payload = self._request("GET", path, params=params)
            
            # Extract data from response
            # Response structure: data[] contains metric objects
            # Each metric has: metric, results[], and each result has: name, columns{}, data[]
            # Data points are arrays: [timestamp, value]
            data = payload.get("data", [])
            
            # Handle response structure
            for metric_obj in data:
                if not isinstance(metric_obj, dict):
                    continue
                
                metric_name = metric_obj.get("metric")
                results = metric_obj.get("results", [])
                
                # Process each result (which may be grouped by region, etc.)
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    
                    # Get grouping information (e.g., region)
                    columns = result.get("columns", {})
                    result_name = result.get("name", "")
                    
                    # Get timeseries data points - array of [timestamp, value]
                    timeseries_data = result.get("data", [])
                    
                    # Convert array format [timestamp, value] to dict format
                    for point in timeseries_data:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            timestamp = point[0]
                            value = point[1]
                            
                            # Create record in expected format
                            record = {
                                "timestamp": timestamp,
                                "value": value,
                                "metric": metric_name,
                            }
                            
                            # Add grouping columns (e.g., region, network_region)
                            record.update(columns)
                            
                            # Add result name for reference
                            if result_name:
                                record["result_name"] = result_name
                            
                            yield record
                        elif isinstance(point, dict):
                            # If already a dict, just add metric and yield
                            point_record = point.copy()
                            if metric_name and "metric" not in point_record:
                                point_record["metric"] = metric_name
                            point_record.update(columns)
                            yield point_record
            
        except requests.HTTPError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "Network data not available for network '%s' with metrics %s (404)",
                    network_code, metrics
                )
                return
            raise