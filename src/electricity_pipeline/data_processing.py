from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)

TIMESTAMP_FIELDS = ("timestamp", "settlement_date", "interval_start")
VALUE_FIELDS = ("value", "power", "emissions", "emission", "co2")


@dataclass
class FacilityRecord:
    facility_id: str
    name: Optional[str]
    fuel_type: Optional[str]
    network_region: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]


def _extract_timestamp(record: Dict) -> datetime:
    for field in TIMESTAMP_FIELDS:
        if field in record:
            raw_value = record[field]
            if isinstance(raw_value, datetime):
                return raw_value
            return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00"))
    raise KeyError(f"No timestamp field found in record: {json.dumps(record)}")


def _extract_value(record: Dict) -> float:
    for field in VALUE_FIELDS:
        if field in record:
            try:
                return float(record[field])
            except (TypeError, ValueError):
                logger.warning("Invalid numeric value for %s in %s", field, record)
                return float("nan")
    raise KeyError(f"No value field found in record: {json.dumps(record)}")


def normalise_timeseries_records(records: Iterator[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for record in records:
        try:
            timestamp = _extract_timestamp(record)
            value = _extract_value(record)
        except KeyError as exc:
            logger.warning("Skipping malformed record: %s", exc)
            continue
        
        # Build normalized record
        normalized = {
            "metric": record.get("metric"),
            "timestamp": timestamp,
            "value": value,
        }
        
        # Add facility_id if present (for facility-level data)
        # Network-level data may not have facility_id
        if "facility_id" in record:
            normalized["facility_id"] = record.get("facility_id")
        
        # Preserve additional fields like region, network_region for network data
        if "region" in record:
            normalized["region"] = record.get("region")
        if "network_region" in record:
            normalized["network_region"] = record.get("network_region")
        
        rows.append(normalized)
    
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No records normalised from API payload.")
        return df
    
    # Drop records missing required fields (metric and timestamp are always required)
    # facility_id is optional (for network-level data)
    required_fields = ["metric", "timestamp"]
    df = df.dropna(subset=required_fields)
    return df


def pivot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    # Determine index columns - use facility_id if present, otherwise use region/network_region
    index_cols = ["timestamp"]
    if "facility_id" in df.columns:
        index_cols.insert(0, "facility_id")
    elif "region" in df.columns:
        index_cols.insert(0, "region")
    elif "network_region" in df.columns:
        index_cols.insert(0, "network_region")
    
    pivot = (
        df.pivot_table(
            index=index_cols,
            columns="metric",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("timestamp")
    )
    pivot.columns.name = None
    return pivot


def load_facility_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Facility metadata not found at {path}. Please provide Assignment 1 output."
        )
    df = pd.read_csv(path)
    required_columns = {"facility_id", "name", "latitude", "longitude"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Facility metadata missing required columns: {missing}")
    return df


def merge_with_metadata(
    consolidated: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    if consolidated.empty:
        return consolidated
    merged = consolidated.merge(metadata, on="facility_id", how="left")
    numeric_cols = merged.select_dtypes(include=["float64", "float32", "int64"]).columns
    merged[numeric_cols] = merged[numeric_cols].interpolate(limit_direction="both")
    return merged


def filter_by_optional_metrics(
    df: pd.DataFrame, optional_metrics: Iterable[str]
) -> pd.DataFrame:
    base_metrics = [col for col in ["power", "emissions"] if col in df.columns]
    available_columns = [
        col for col in optional_metrics if col in df.columns and df[col].notna().any()
    ]
    keep_columns = [
        "timestamp",
        *base_metrics,
        *available_columns,
    ]
    
    # Add facility_id if present (for facility-level data)
    if "facility_id" in df.columns:
        keep_columns.insert(0, "facility_id")
    # Add region if present (for network-level data)
    elif "region" in df.columns:
        keep_columns.insert(0, "region")
    elif "network_region" in df.columns:
        keep_columns.insert(0, "network_region")
    
    keep_columns.extend(
        col
        for col in ["name", "fuel_type", "network_region", "latitude", "longitude"]
        if col in df.columns
    )
    # Use list(set()) to deduplicate while preserving order
    return df.loc[:, list(dict.fromkeys(keep_columns))]
