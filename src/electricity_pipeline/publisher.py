from __future__ import annotations

import json
import logging
import time
from typing import Iterable, Optional

import pandas as pd
import paho.mqtt.client as mqtt

from .config import MqttConfig


logger = logging.getLogger(__name__)


def _clean_value(value):
    if pd.isna(value):
        return None
    return value


def _build_payload(row: pd.Series) -> str:
    # Handle both facility-level and network-level data
    identifier_key = "facility_id" if "facility_id" in row else "region" if "region" in row else "network_region"
    identifier_value = row.get(identifier_key, "NETWORK")
    
    payload = {
        identifier_key: identifier_value,
        "timestamp": row["timestamp"].isoformat()
        if hasattr(row["timestamp"], "isoformat")
        else str(row["timestamp"]),
        "power": _clean_value(row.get("power")),
        "emissions": _clean_value(row.get("emissions")),
        "metadata": {
            "name": _clean_value(row.get("name")),
            "fuel_type": _clean_value(row.get("fuel_type")),
            "network_region": _clean_value(row.get("network_region") or row.get("region")),
            "latitude": _clean_value(row.get("latitude")),
            "longitude": _clean_value(row.get("longitude")),
        },
    }
    for optional_field in ["price", "demand"]:
        if optional_field in row and pd.notna(row[optional_field]):
            payload[optional_field] = _clean_value(row[optional_field])
    return json.dumps(payload)


def publish_dataset(
    df: pd.DataFrame,
    config: MqttConfig,
    delay_seconds: float = 0.1,
    topic_override: Optional[str] = None,
) -> None:
    if df.empty:
        logger.warning("No data to publish to MQTT.")
        return

    client = mqtt.Client()
    if config.username and config.password:
        client.username_pw_set(config.username, config.password)

    logger.info("Connecting to MQTT broker %s:%s", config.host, config.port)
    client.connect(config.host, config.port, config.keepalive)
    client.loop_start()

    topic = topic_override or config.topic
    for _, row in df.sort_values("timestamp").iterrows():
        payload = _build_payload(row)
        result = client.publish(topic, payload, qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error("Failed to publish message: %s", mqtt.error_string(result.rc))
        logger.debug("Published message to %s: %s", topic, payload)
        time.sleep(delay_seconds)

    client.loop_stop()
    client.disconnect()
