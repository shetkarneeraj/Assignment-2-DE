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


logger = logging.getLogger(__name__)


def retrieve_and_cache_dataset(
    config: PipelineConfig,
    start: datetime,
    end: datetime,
    metrics: Sequence[str],
    use_cache: bool = True,
    allow_partial: bool = False,
    interval: str = "5m",
    network_region: Optional[str] = None,
    primary_grouping: Optional[str] = None,
    use_facility_endpoint: bool = False,
) -> pd.DataFrame:
    """
    Retrieve and cache dataset from OpenElectricity API.
    
    Can use either:
    - Network endpoint: GET /v4/market/network/{network_code} (network-level aggregated data)
    - Facility endpoint: GET /v4/facilities/{facility_code}/metrics/{metric} (per-facility data)
    
    Args:
        config: Pipeline configuration
        start: Start datetime
        end: End datetime
        metrics: List of metrics to retrieve (e.g., ["power", "emissions", "price", "demand"])
        use_cache: Whether to use cached data if available
        allow_partial: Whether to allow empty datasets
        interval: Time interval (5m, 1h, 1d, etc.)
        network_region: Optional network region filter (for network endpoint)
        primary_grouping: Optional grouping (for network endpoint)
        use_facility_endpoint: If True, fetch per-facility data instead of network-level
    """
    cache_path = build_cache_path(config.cache, start, end)
    if use_cache and cache_path.exists():
        return read_cached_dataset(cache_path)

    client = OpenElectricityClient(config.api)
    
    if use_facility_endpoint:
        # Fetch per-facility data
        logger.info(
            "Retrieving metrics %s per facility for network %s",
            metrics, config.api.network
        )
        
        # Separate facility-level and network-level metrics
        facility_level_metrics = [m for m in metrics if m not in ["price", "demand"]]
        network_level_metrics = [m for m in metrics if m in ["price", "demand"]]
        
        # Create iterator that yields from both sources
        def combined_records():
            # Fetch facility-level metrics (power, emissions)
            if facility_level_metrics:
                # Get all facilities first
                facilities = client.fetch_facilities()
                
                # Get metric names from config
                facility_metrics = [config.api.metrics.get(metric, metric) for metric in facility_level_metrics]
                
                # Fetch metrics for each facility
                facility_records = client.fetch_facilities_metrics(
                    facilities=facilities,
                    metrics=facility_metrics,
                    start=start,
                    end=end,
                    interval=interval,
                )
                yield from facility_records
            
            # Fetch network-level metrics (price, demand) from network endpoint
            if network_level_metrics:
                logger.info(
                    "Fetching network-level metrics %s from network endpoint (not available per-facility)",
                    network_level_metrics
                )
                network_metrics = [config.api.metrics.get(metric, metric) for metric in network_level_metrics]
                network_records = client.fetch_network_data(
                    metrics=network_metrics,
                    start=start,
                    end=end,
                    interval=interval,
                    network_code=config.api.network,
                    network_region=network_region,
                    primary_grouping=primary_grouping,
                )
                yield from network_records
        
        # Normalize all records
        normalised = normalise_timeseries_records(combined_records())
        
        if normalised.empty:
            if allow_partial:
                logger.warning("No data retrieved; returning empty DataFrame.")
                return normalised
            raise RuntimeError("No data retrieved from OpenElectricity API.")

        # Separate facility-level and network-level data for proper pivoting
        # Network data has price/demand metrics but no facility_id
        # Facility data has facility_id and power/emissions
        has_facility_id = "facility_id" in normalised.columns
        if has_facility_id:
            # Facility data: has facility_id (not null/NaN)
            facility_df = normalised[normalised["facility_id"].notna()].copy()
            # Network data: no facility_id OR facility_id is null/NaN
            network_df = normalised[
                (normalised["facility_id"].isna()) | 
                (normalised["metric"].isin(["price", "demand"]))
            ].copy()
            # Remove facility_id from network data since it's not relevant
            if "facility_id" in network_df.columns:
                network_df = network_df.drop(columns=["facility_id"], errors="ignore")
            
            # Pivot facility data
            if not facility_df.empty:
                facility_pivoted = pivot_metrics(facility_df)
            else:
                facility_pivoted = pd.DataFrame()
            
            # Pivot network data separately (no facility_id)
            if not network_df.empty:
                # For network data, pivot on timestamp only
                network_pivoted = (
                    network_df.pivot_table(
                        index=["timestamp"],
                        columns="metric",
                        values="value",
                        aggfunc="mean",
                    )
                    .reset_index()
                    .sort_values("timestamp")
                )
                network_pivoted.columns.name = None
                
                # Merge network data with facility data
                # For each facility row, add the network-level price/demand
                if not facility_pivoted.empty and not network_pivoted.empty:
                    # Merge on timestamp - this will add price/demand to all facility rows
                    consolidated = facility_pivoted.merge(
                        network_pivoted[["timestamp"] + [col for col in network_pivoted.columns if col in ["price", "demand"]]],
                        on="timestamp",
                        how="left"
                    )
                else:
                    consolidated = facility_pivoted if not facility_pivoted.empty else network_pivoted
            else:
                consolidated = facility_pivoted
        else:
            # No facility_id at all - just pivot normally
            consolidated = pivot_metrics(normalised)
        
        # Merge with facility metadata
        try:
            metadata = load_facility_metadata(config.facilities_metadata_path)
            merged = merge_with_metadata(consolidated, metadata)
        except FileNotFoundError:
            logger.warning("Facility metadata file not found, skipping merge")
            merged = consolidated
        
        cleaned = filter_by_optional_metrics(merged, config.consolidate_optional_metrics)
        
    else:
        # Use network endpoint (existing behavior)
        logger.info(
            "Retrieving metrics %s from network endpoint for network %s",
            metrics, config.api.network
        )
        
        # Get network-level metrics
        network_metrics = [config.api.metrics.get(metric, metric) for metric in metrics]
        records = client.fetch_network_data(
            metrics=network_metrics,
            start=start,
            end=end,
            interval=interval,
            network_code=config.api.network,
            network_region=network_region,
            primary_grouping=primary_grouping,
        )
        
        # Normalize network data records
        normalised = normalise_timeseries_records(records)
        
        if normalised.empty:
            if allow_partial:
                logger.warning("No data retrieved; returning empty DataFrame.")
                return normalised
            raise RuntimeError("No data retrieved from OpenElectricity API.")

        consolidated = pivot_metrics(normalised)
        
        # Network data doesn't need facility metadata
        cleaned = filter_by_optional_metrics(consolidated, config.consolidate_optional_metrics)
    
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
