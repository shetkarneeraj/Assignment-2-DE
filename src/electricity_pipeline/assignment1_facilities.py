from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import requests
from urllib.parse import quote


logger = logging.getLogger(__name__)


STATE_TO_REGION = {
    "NSW": "NSW1",
    "VIC": "VIC1",
    "QLD": "QLD1",
    "SA": "SA1",
    "TAS": "TAS1",
}


DEFAULT_USER_AGENT = "Assignment2ElectricityPipeline/1.0 (+contact@example.com)"


@dataclass
class Assignment1DataConfig:
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    metadata_output: Path = Path("data/facilities_metadata.csv")
    geocode_cache_file: Path = Path("cache/geocode_cache.json")
    min_capacity_mw: float = 10.0
    geocode_delay_seconds: float = 1.0
    user_agent: str = DEFAULT_USER_AGENT


def slugify(value: str) -> str:
    value = value or ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _standardise_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).title()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Assignment 1 dataset missing: {path}")
    return pd.read_csv(path)


def load_raw_cer_tables(config: Assignment1DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probable = _load_csv(config.data_dir / "power-stations-and-projects-probable.csv")
    committed = _load_csv(config.data_dir / "power-stations-and-projects-committed.csv")
    accredited = _load_csv(config.data_dir / "power-stations-and-projects-accredited.csv")
    return probable, committed, accredited


def _rename_columns(probable: pd.DataFrame, committed: pd.DataFrame, accredited: pd.DataFrame) -> None:
    probable.rename(
        columns={
            "Project Name": "Name",
            "State ": "State",
            "MW Capacity": "Capacity",
            "Fuel Source": "Fuel",
        },
        inplace=True,
    )
    committed.rename(
        columns={
            "Project Name": "Name",
            "State ": "State",
            "MW Capacity": "Capacity",
            "Fuel Source": "Fuel",
            "Committed Date (Month/Year)": "Committed_Date",
        },
        inplace=True,
    )
    accredited.rename(
        columns={
            "Power station name": "Name",
            "State": "State",
            "Installed capacity (MW)": "Capacity",
            "Fuel Source (s)": "Fuel",
            "Accreditation start date": "Accreditation_Start",
            "Approval date": "Approval_Date",
            "Accreditation code": "Accreditation_Code",
        },
        inplace=True,
    )


def _clean_tables(probable: pd.DataFrame, committed: pd.DataFrame, accredited: pd.DataFrame) -> pd.DataFrame:
    probable["Status"] = "Probable"
    committed["Status"] = "Committed"
    accredited["Status"] = "Accredited"

    frames = []
    for df in (probable, committed, accredited):
        df["Name"] = df["Name"].astype(str)
        df["Name_clean"] = df["Name"].apply(_standardise_name)
        df["State"] = df["State"].astype(str).str.strip().str.upper()
        if "Postcode" in df.columns:
            df["Postcode"] = df["Postcode"].astype(str).str.strip()
        for date_col in ("Committed_Date", "Accreditation_Start", "Approval_Date"):
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")
        frames.append(
            df[
                [
                    col
                    for col in [
                        "Name",
                        "Name_clean",
                        "State",
                        "Capacity",
                        "Fuel",
                        "Status",
                        "Postcode",
                        "Committed_Date",
                        "Accreditation_Start",
                        "Approval_Date",
                        "Accreditation_Code",
                    ]
                    if col in df.columns
                ]
            ].copy()
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["Capacity"] = pd.to_numeric(combined["Capacity"], errors="coerce")
    combined = combined.dropna(subset=["Capacity"])
    combined = combined[combined["Capacity"] >= 0]
    return combined


def _load_geocode_cache(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Geocode cache at %s is corrupt; starting fresh.", path)
        return {}


def _save_geocode_cache(path: Path, cache: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _geocode_query(query: str, user_agent: str) -> Optional[Tuple[float, float]]:
    url = f"https://nominatim.openstreetmap.org/search?q={quote(query)}&format=json&limit=1&countrycodes=au"
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return None
    result = payload[0]
    return float(result["lat"]), float(result["lon"])


def geocode_with_fallbacks(
    name: str,
    state: str,
    postcode: Optional[str],
    *,
    cache: Dict[str, Dict[str, float]],
    config: Assignment1DataConfig,
) -> Tuple[Optional[float], Optional[float]]:
    cache_key = "|".join(filter(None, [name or "", state or "", postcode or ""]))
    if cache_key in cache:
        entry = cache[cache_key]
        return entry.get("lat"), entry.get("lon")

    queries = [
        f"{name} {state} Australia",
        f"{name.split(' ')[0]} {state} Australia",
    ]
    if postcode:
        queries.append(f"{name} {postcode} Australia")
        queries.append(f"Australia postcode {postcode}")

    for idx, query in enumerate(queries):
        try:
            lat_lon = _geocode_query(query, config.user_agent)
        except requests.RequestException as exc:
            logger.debug("Geocoding query '%s' failed: %s", query, exc)
            lat_lon = None

        if lat_lon:
            cache[cache_key] = {"lat": lat_lon[0], "lon": lat_lon[1]}
            _save_geocode_cache(config.geocode_cache_file, cache)
            return lat_lon
        if idx < len(queries) - 1:
            time.sleep(config.geocode_delay_seconds)
    cache[cache_key] = {"lat": None, "lon": None}
    _save_geocode_cache(config.geocode_cache_file, cache)
    return None, None


def fill_missing_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    state_centroids = df.groupby("State")[["lat", "lon"]].median(numeric_only=True)
    missing_mask = df["lat"].isna() | df["lon"].isna()
    for idx in df[missing_mask].index:
        state = df.at[idx, "State"]
        if state in state_centroids.index:
            df.at[idx, "lat"] = state_centroids.loc[state, "lat"]
            df.at[idx, "lon"] = state_centroids.loc[state, "lon"]
    return df


def attach_geocodes(df: pd.DataFrame, config: Assignment1DataConfig) -> pd.DataFrame:
    cache = _load_geocode_cache(config.geocode_cache_file)
    df["lat"] = pd.NA
    df["lon"] = pd.NA
    for idx, row in df.iterrows():
        lat, lon = geocode_with_fallbacks(
            row["Name_clean"],
            row["State"],
            row.get("Postcode"),
            cache=cache,
            config=config,
        )
        df.at[idx, "lat"] = lat
        df.at[idx, "lon"] = lon
    df = fill_missing_coordinates(df)
    return df


def build_facilities_metadata(config: Assignment1DataConfig, *, skip_geocode: bool = False) -> pd.DataFrame:
    probable, committed, accredited = load_raw_cer_tables(config)
    _rename_columns(probable, committed, accredited)
    combined = _clean_tables(probable, committed, accredited)
    combined = combined[combined["Capacity"] >= config.min_capacity_mw].reset_index(drop=True)

    if not skip_geocode:
        logger.info("Geocoding %d facilities (may take several minutes)...", len(combined))
        combined = attach_geocodes(combined, config)
    else:
        logger.warning("Skipping geocoding; latitude/longitude may be empty.")

    combined["facility_id"] = combined.apply(
        lambda row: slugify(f"{row['Name_clean']}_{row['State']}"), axis=1
    )
    combined["fuel_type"] = combined["Fuel"].str.strip()
    combined["network_region"] = combined["State"].map(STATE_TO_REGION)
    combined.rename(columns={"lat": "latitude", "lon": "longitude", "State": "state"}, inplace=True)

    columns = [
        "facility_id",
        "Name",
        "Name_clean",
        "state",
        "network_region",
        "fuel_type",
        "Status",
        "Capacity",
        "latitude",
        "longitude",
    ]
    optional_cols = [col for col in columns if col in combined.columns]
    metadata = combined[optional_cols].copy()
    metadata.rename(
        columns={
            "Name": "name",
            "Name_clean": "name_clean",
            "Status": "status",
            "Capacity": "capacity_mw",
        },
        inplace=True,
    )
    metadata = metadata.drop_duplicates(subset=["facility_id"])
    metadata = metadata.reset_index(drop=True)
    metadata["name_key"] = metadata["name_clean"].apply(slugify)
    return metadata


def write_metadata(metadata: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(path, index=False)
    logger.info("Wrote facilities metadata to %s (%d records)", path, len(metadata))


def build_and_store_metadata(config: Assignment1DataConfig, *, skip_geocode: bool = False) -> pd.DataFrame:
    metadata = build_facilities_metadata(config, skip_geocode=skip_geocode)
    write_metadata(metadata, config.metadata_output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate facilities metadata from Assignment 1 datasets."
    )
    parser.add_argument(
        "--skip-geocode",
        action="store_true",
        help="Skip live geocoding (uses cached/state centroids only).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Assignment1DataConfig().data_dir,
        help="Directory that contains Assignment 1 CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Assignment1DataConfig().metadata_output,
        help="Output CSV path for facilities metadata.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    config = Assignment1DataConfig(data_dir=args.data_dir, metadata_output=args.output)
    build_and_store_metadata(config, skip_geocode=args.skip_geocode)


if __name__ == "__main__":
    main()

