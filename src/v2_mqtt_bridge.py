from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from urllib import request

from .config import (
    API_PREDICT_EVENTS_URL,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_SENSOR_TOPIC,
)
from .logging_utils import configure_logging, get_logger


logger = get_logger(__name__)


def _load_mqtt_module():
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError(
            "paho-mqtt is required for MQTT bridge. Install requirements.txt or run through Docker Compose."
        ) from exc
    return mqtt


def _post_events(api_url: str, events: list[dict[str, object]]) -> None:
    payload = json.dumps({"events": events}).encode("utf-8")
    req = request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        response.read()


def run_bridge(
    broker_host: str = MQTT_BROKER_HOST,
    broker_port: int = MQTT_BROKER_PORT,
    topic: str = MQTT_SENSOR_TOPIC,
    api_url: str = API_PREDICT_EVENTS_URL,
    batch_size: int = 10,
    flush_seconds: float = 1.0,
) -> None:
    mqtt = _load_mqtt_module()
    event_queue: queue.Queue[dict[str, object]] = queue.Queue()
    stop_event = threading.Event()

    def on_connect(client, _userdata, _flags, reason_code, _properties=None):
        logger.info("mqtt_bridge_connected", extra={"reason_code": str(reason_code), "topic": topic})
        client.subscribe(topic)

    def on_message(_client, _userdata, message):
        try:
            event_queue.put(json.loads(message.payload.decode("utf-8")))
        except json.JSONDecodeError:
            logger.warning("mqtt_bridge_bad_payload", extra={"topic": message.topic})

    def forward_loop() -> None:
        pending: list[dict[str, object]] = []
        last_flush = time.monotonic()
        while not stop_event.is_set():
            try:
                pending.append(event_queue.get(timeout=0.2))
            except queue.Empty:
                pass

            should_flush = pending and (
                len(pending) >= batch_size or time.monotonic() - last_flush >= flush_seconds
            )
            if should_flush:
                try:
                    _post_events(api_url, pending)
                    logger.info("mqtt_bridge_forwarded", extra={"events": len(pending), "api_url": api_url})
                    pending = []
                    last_flush = time.monotonic()
                except OSError as exc:
                    logger.warning("mqtt_bridge_forward_failed", extra={"error": str(exc)})
                    time.sleep(1)

    forwarder = threading.Thread(target=forward_loop, daemon=True)
    forwarder.start()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_host, broker_port, keepalive=30)

    try:
        client.loop_forever()
    finally:
        stop_event.set()
        client.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge MQTT sensor events into the FastAPI prediction endpoint.")
    parser.add_argument("--broker-host", default=MQTT_BROKER_HOST)
    parser.add_argument("--broker-port", type=int, default=MQTT_BROKER_PORT)
    parser.add_argument("--topic", default=MQTT_SENSOR_TOPIC)
    parser.add_argument("--api-url", default=API_PREDICT_EVENTS_URL)
    parser.add_argument("--batch-size", type=int, default=10)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    run_bridge(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        topic=args.topic,
        api_url=args.api_url,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
