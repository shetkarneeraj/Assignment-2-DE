from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
import paho.mqtt.client as mqtt

from .config import MqttConfig


logger = logging.getLogger(__name__)


def _parse_message(payload: str) -> Optional[dict]:
    try:
        message = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Discarding invalid JSON payload: %s", payload)
        return None
    if "timestamp" in message:
        try:
            message["timestamp"] = datetime.fromisoformat(
                message["timestamp"].replace("Z", "+00:00")
            )
        except ValueError:
            logger.warning("Invalid timestamp in message: %s", payload)
            return None
    return message


@dataclass
class MessageStore:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _data: Optional[pd.DataFrame] = field(default=None, init=False)

    def append(self, message: dict) -> None:
        with self._lock:
            row = pd.DataFrame([message])
            if self._data is None:
                self._data = row
            else:
                self._data = pd.concat([self._data, row], ignore_index=True)

    def snapshot(self) -> pd.DataFrame:
        with self._lock:
            if self._data is None:
                return pd.DataFrame()
            return self._data.copy()


class MqttSubscriber:
    def __init__(
        self,
        config: MqttConfig,
        topic: Optional[str] = None,
        on_message: Optional[Callable[[dict], None]] = None,
    ):
        self.config = config
        self.topic = topic or config.topic
        self.store = MessageStore()
        self.on_message = on_message or self.store.append
        self.client = mqtt.Client()
        if config.username and config.password:
            self.client.username_pw_set(config.username, config.password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker, subscribing to %s", self.topic)
            client.subscribe(self.topic, qos=1)
        else:
            logger.error("Failed to connect to MQTT broker: %s", rc)

    def _on_message(self, client, userdata, msg):
        message = _parse_message(msg.payload.decode("utf-8"))
        if message is not None:
            self.on_message(message)

    def start(self) -> None:
        logger.info("Starting MQTT subscriber loop")
        self.client.connect(self.config.host, self.config.port, self.config.keepalive)
        self.client.loop_start()

    def stop(self) -> None:
        logger.info("Stopping MQTT subscriber loop")
        self.client.loop_stop()
        self.client.disconnect()

