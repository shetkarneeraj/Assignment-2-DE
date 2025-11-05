from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd

from .assignment1_facilities import (
    Assignment1DataConfig,
    build_and_store_metadata,
    slugify,
)


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

        normalized = {
            "metric": record.get("metric"),
            "timestamp": timestamp,
            "value": value,
        }

        if "facility_id" in record:
            normalized["facility_id"] = record.get("facility_id")

        if "region" in record:
            normalized["region"] = record.get("region")
        if "network_region" in record:
            normalized["network_region"] = record.get("network_region")

        rows.append(normalized)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No records normalised from API payload.")
        return df

    required_fields = ["metric", "timestamp"]
    df = df.dropna(subset=required_fields)
    return df


def pivot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

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


def _ensure_metadata_file(path: Path) -> None:
    if path.exists():
        return
    logger.info("Facility metadata missing at %s. Building from Assignment 1 data.", path)
    config = Assignment1DataConfig(
        data_dir=path.parent,
        metadata_output=path,
    )
    try:
        build_and_store_metadata(config, skip_geocode=False)
    except Exception as exc:
        logger.warning(
            "Live geocoding failed (%s). Retrying using cached/state centroids.", exc
        )
        build_and_store_metadata(config, skip_geocode=True)


def load_facility_metadata(path: Path, auto_build: bool = True) -> pd.DataFrame:
    if not path.exists():
        if not auto_build:
            raise FileNotFoundError(
                f"Facility metadata not found at {path}. Please provide Assignment 1 output."
            )
        try:
            _ensure_metadata_file(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Assignment 1 datasets not found in {path.parent}. "
                "Ensure the CER CSV files are available."
            ) from exc
    df = pd.read_csv(path)
    required_columns = {"facility_id", "name", "latitude", "longitude"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Facility metadata missing required columns: {missing}")
    if "name_clean" not in df.columns:
        df["name_clean"] = df["name"].astype(str).str.strip().str.title()
    if "name_key" not in df.columns:
        df["name_key"] = df["name_clean"].apply(slugify)
    return df


def _derive_name_key(df: pd.DataFrame) -> pd.Series:
    name_source = None
    for candidate in ("name", "facility_name"):
        if candidate in df.columns:
            name_source = df[candidate]
            break
    if name_source is None and "facility_id" in df.columns:
        name_source = df["facility_id"]
    if name_source is None:
        name_source = pd.Series(index=df.index, dtype=str)
    return name_source.fillna("").astype(str).apply(slugify)


def merge_with_metadata(
    consolidated: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    if consolidated.empty:
        return consolidated

    metadata = metadata.copy()
    if "name_key" not in metadata.columns:
        metadata["name_key"] = metadata["name"].astype(str).apply(slugify)
    metadata_name_lookup = (
        metadata.drop_duplicates(subset=["name_key"], keep="first")
        .set_index("name_key")
    )

    merged = consolidated.merge(
        metadata.drop(columns=["name_key"]),
        on="facility_id",
        how="left",
        suffixes=("", "_meta"),
    )
    merged["name_key"] = _derive_name_key(merged)

    missing_mask = merged["latitude"].isna() & merged["name_key"].notna()
    if missing_mask.any():
        fallback = metadata_name_lookup.reindex(merged.loc[missing_mask, "name_key"])
        for column in ["name", "fuel_type", "network_region", "latitude", "longitude"]:
            if column in fallback.columns:
                merged.loc[missing_mask, column] = merged.loc[missing_mask, column].fillna(
                    fallback[column].values
                )

    for api_col, target_col in [
        ("facility_name", "name"),
        ("facility_fuel_type", "fuel_type"),
        ("facility_network_region", "network_region"),
        ("facility_latitude", "latitude"),
        ("facility_longitude", "longitude"),
    ]:
        if api_col in merged.columns:
            merged[target_col] = merged[target_col].fillna(merged[api_col])

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
        "facility_id",
        "timestamp",
        *base_metrics,
        *available_columns,
    ]

    if "region" in df.columns and "facility_id" not in df.columns:
        keep_columns.insert(0, "region")
    elif "network_region" in df.columns and "facility_id" not in df.columns:
        keep_columns.insert(0, "network_region")

    keep_columns.extend(
        col
        for col in ["name", "fuel_type", "network_region", "latitude", "longitude"]
        if col in df.columns
    )
    return df.loc[:, list(dict.fromkeys(keep_columns))]
