import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml


def _find_config_path() -> Path:
    """Find config.yml by checking current dir, parent dirs, and package root."""
    # Try current working directory
    cwd_config = Path.cwd() / "config.yml"
    if cwd_config.exists():
        return cwd_config
    
    # Try parent directories (up to 3 levels)
    current = Path.cwd()
    for _ in range(3):
        parent_config = current.parent / "config.yml"
        if parent_config.exists():
            return parent_config
        current = current.parent
    
    # Try relative to this file's directory
    module_dir = Path(__file__).parent
    project_root = module_dir.parent.parent
    root_config = project_root / "config.yml"
    if root_config.exists():
        return root_config
    
    # Default fallback
    return Path("config.yml")


DEFAULT_CONFIG_PATH = _find_config_path()


@dataclass
class ApiConfig:
    base_url: str = "https://api.openelectricity.org.au/v4"
    api_key: Optional[str] = None
    network: str = "NEM"
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    allow_parallel_requests: bool = False
    metrics: Dict[str, str] = field(
        default_factory=lambda: {
            "power": "power",
            "emissions": "emissions",
            "price": "price",
            "demand": "demand",
        }
    )
    facility_endpoint: str = "/facilities"
    timeseries_endpoint: str = "/facilities/{facility_id}/metrics/{metric}"
    max_page_size: int = 288  # 5-minute data for 24 hours


@dataclass
class CacheConfig:
    directory: Path = Path("cache")
    consolidated_filename: str = "consolidated.csv"


@dataclass
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    topic: str = "openelectricity/facilities"
    username: Optional[str] = None
    password: Optional[str] = None
    keepalive: int = 60


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8050
    refresh_interval_ms: int = 2000
    mapbox_token: Optional[str] = None


@dataclass
class PipelineConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    facilities_metadata_path: Path = Path("data/facilities_metadata.csv")
    consolidate_optional_metrics: Iterable[str] = field(
        default_factory=lambda: ["price", "demand"]
    )


def _merge_dicts(base: Dict, update: Dict) -> Dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    config_dict = {}

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

    api_key = (
        os.getenv("OPEN_ELECTRICITY_API_KEY")
        or os.getenv("OPEN_ELECTRICITY_AUTH_TOKEN")
        or os.getenv("OPEN_ELECTRICITY_TOKEN")
    )
    if api_key:
        config_dict = _merge_dicts(
            config_dict,
            {"api": {"api_key": api_key.strip()}},
        )

    mqtt_password = os.getenv("MQTT_PASSWORD")
    mqtt_username = os.getenv("MQTT_USERNAME")
    mqtt_overrides = {}
    if mqtt_password:
        mqtt_overrides.setdefault("mqtt", {})["password"] = mqtt_password
    if mqtt_username:
        mqtt_overrides.setdefault("mqtt", {})["username"] = mqtt_username
    if mqtt_overrides:
        config_dict = _merge_dicts(config_dict, mqtt_overrides)

    # Convert cache directory to Path if it's a string
    cache_config = config_dict.get("cache", {})
    if "directory" in cache_config and isinstance(cache_config["directory"], str):
        cache_config["directory"] = Path(cache_config["directory"])

    return PipelineConfig(
        api=ApiConfig(**config_dict.get("api", {})),
        cache=CacheConfig(**cache_config),
        mqtt=MqttConfig(**config_dict.get("mqtt", {})),
        dashboard=DashboardConfig(**config_dict.get("dashboard", {})),
        facilities_metadata_path=Path(
            config_dict.get("facilities_metadata_path", "data/facilities_metadata.csv")
        ),
        consolidate_optional_metrics=config_dict.get(
            "consolidate_optional_metrics", ["price", "demand"]
        ),
    )
