from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence

import pandas as pd

from .api_client import OpenElectricityClient
from .caching import build_cache_path, read_cached_dataset, write_dataset_to_cache
from .config import PipelineConfig
from .data_processing import (
    filter_by_optional_metrics,
    load_facility_metadata,
    merge_with_metadata,
    normalise_timeseries_records,
    pivot_metrics,
)
from .assignment1_facilities import slugify


logger = logging.getLogger(__name__)


def _normalise_facility_catalog(facilities: Iterable[dict]) -> pd.DataFrame:
    records = []
    for facility in facilities:
        facility_id = (
            facility.get("code")
            or facility.get("id")
            or facility.get("facility_id")
        )
        if not facility_id:
            continue
        name = facility.get("name") or facility.get("label")
        # Extract coordinates from location field if available
        latitude = facility.get("latitude")
        longitude = facility.get("longitude")
        if not latitude and "location" in facility and facility["location"]:
            latitude = facility["location"].get("lat")
            longitude = facility["location"].get("lng")

        record = {
            "facility_id": facility_id,
            "facility_name": name,
            "facility_state": facility.get("state") or facility.get("state_id"),
            "facility_network_region": facility.get("network_region")
            or facility.get("network_region_id"),
            "facility_fuel_type": facility.get("fueltech_id")
            or facility.get("fueltech_code")
            or facility.get("fueltech"),
            "facility_latitude": latitude,
            "facility_longitude": longitude,
        }
        if name:
            record["name_key"] = slugify(name)
        records.append(record)
    return pd.DataFrame.from_records(records)


def retrieve_and_cache_dataset(
    config: PipelineConfig,
    start: datetime,
    end: datetime,
    metrics: Sequence[str],
    use_cache: bool = True,
    allow_partial: bool = False,
    interval: str = "PT5M",
    **_: object,
) -> pd.DataFrame:
    """
    Retrieve facility-level metrics, merge with Assignment 1 metadata, and cache the result.
    Additional kwargs are accepted for backward compatibility but ignored.
    """
    cache_path = build_cache_path(config.cache, start, end)
    if use_cache and cache_path.exists():
        return read_cached_dataset(cache_path)

    client = OpenElectricityClient(config.api)
    facilities = client.fetch_facilities()
    facility_catalog = _normalise_facility_catalog(facilities)
    metric_pairs = [
        (metric, config.api.metrics.get(metric, metric)) for metric in metrics
    ]
    logger.info(
        "Retrieving metrics %s for %d facilities", metric_pairs, len(facilities)
    )

    records = client.fetch_metrics_for_facilities(
        facilities=facilities,
        metrics=metric_pairs,
        start=start,
        end=end,
        interval=interval,
    )
    normalised = normalise_timeseries_records(records)
    if normalised.empty:
        if allow_partial:
            logger.warning("No data retrieved; returning empty DataFrame.")
            return normalised
        raise RuntimeError("No data retrieved from OpenElectricity API.")

    consolidated = pivot_metrics(normalised)
    if not facility_catalog.empty:
        consolidated = consolidated.merge(
            facility_catalog, on="facility_id", how="left"
        )

    metadata = load_facility_metadata(config.facilities_metadata_path)
    merged = merge_with_metadata(consolidated, metadata)
    cleaned = filter_by_optional_metrics(merged, config.consolidate_optional_metrics)
    write_dataset_to_cache(cleaned, cache_path)
    return cleaned


def iter_time_windows(
    start: datetime, end: datetime, window: timedelta
) -> Iterable[tuple[datetime, datetime]]:
    current = start
    while current < end:
        window_end = min(current + window, end)
        yield current, window_end
        current = window_end

