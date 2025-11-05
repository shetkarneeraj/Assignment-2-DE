from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Sequence

import pandas as pd

from .config import load_config
from .publisher import publish_dataset
from .retrieval import retrieve_and_cache_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenElectricity data retrieval and MQTT publisher. Uses facility-level endpoints for continuous data streaming."
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="ISO date for the start of the window (e.g. 2025-10-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="ISO date for the end of the window (e.g. 2025-10-08).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["power", "emissions"],
        help="Metrics to retrieve (power emissions price demand).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        help="Time interval (5m, 1h, 1d, etc.). Default: 5m",
    )
    parser.add_argument(
        "--network-region",
        type=str,
        default=None,
        help="Optional network region filter.",
    )
    parser.add_argument(
        "--primary-grouping",
        type=str,
        choices=["network", "network_region"],
        default=None,
        help="Primary grouping (network or network_region).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of iterations to run. 0 means run indefinitely.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=60,
        help="Delay between iterations, defaults to 60 seconds.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Force refresh even if cached data is available.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return empty datasets without raising when API yields no data.",
    )
    parser.add_argument(
        "--use-facility-endpoint",
        action="store_true",
        help="Fetch per-facility data instead of network-level aggregated data.",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    metrics = args.metrics
    config = load_config()

    iteration = 0
    # Task 5: Continuous execution - run indefinitely unless iterations is set
    while True:
        iteration += 1
        logger.info("Starting iteration %d", iteration)
        
        # Task 1-2: Retrieve facility-level power and emissions data
        df = retrieve_and_cache_dataset(
            config=config,
            start=start,
            end=end,
            metrics=metrics,
            use_cache=not args.disable_cache,
            allow_partial=args.allow_partial,
            interval=args.interval,
            network_region=args.network_region,
            primary_grouping=args.primary_grouping,
            use_facility_endpoint=args.use_facility_endpoint,
        )
        
        if df.empty:
            logger.warning("Iteration %d produced no data.", iteration)
        else:
            if "latitude" not in df.columns or df["latitude"].isna().all():
                from .retrieval import _normalise_facility_catalog
                from .api_client import OpenElectricityClient

                client = OpenElectricityClient(config.api)
                facilities = client.fetch_facilities()
                facility_catalog = _normalise_facility_catalog(facilities)

                if not facility_catalog.empty:
                    df = df.merge(
                        facility_catalog[["facility_id", "facility_name", "facility_latitude", "facility_longitude"]],
                        on="facility_id", how="left"
                    )
                    df = df.rename(columns={
                        "facility_name": "name",
                        "facility_latitude": "latitude",
                        "facility_longitude": "longitude"
                    })

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            publish_dataset(df, config.mqtt, delay_seconds=0.1)

        # Task 5: Stop if iterations limit is reached, otherwise continue indefinitely
        if args.iterations and iteration >= args.iterations:
            logger.info("Reached requested iterations, stopping.")
            break
        
        # Task 5: 60-second delay between API data retrieval rounds (in addition to 0.1s delay between messages)
        logger.info("Sleeping for %d seconds before the next iteration.", args.sleep_seconds)
        time.sleep(args.sleep_seconds)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    run_pipeline(args)


if __name__ == "__main__":
    main()
