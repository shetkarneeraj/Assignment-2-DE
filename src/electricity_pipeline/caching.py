from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import CacheConfig


logger = logging.getLogger(__name__)


def ensure_cache_dir(cache_config: CacheConfig) -> Path:
    cache_config.directory.mkdir(parents=True, exist_ok=True)
    return cache_config.directory


def build_cache_path(
    cache_config: CacheConfig,
    start: datetime,
    end: datetime,
    suffix: str = "",
) -> Path:
    ensure_cache_dir(cache_config)
    tag = f"{start.date()}_{end.date()}"
    suffix_part = f"_{suffix}" if suffix else ""
    return cache_config.directory / f"{tag}{suffix_part}_{cache_config.consolidated_filename}"


def read_cached_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    logger.info("Loading cached dataset from %s", path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def write_dataset_to_cache(df: pd.DataFrame, path: Path) -> None:
    ensure_cache_dir(CacheConfig(directory=path.parent, consolidated_filename=path.name))
    df.to_csv(path, index=False)
    logger.info("Stored consolidated dataset at %s", path)

