from __future__ import annotations

import argparse
import json
import time

from .config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_SENSOR_TOPIC, RANDOM_STATE
from .logging_utils import configure_logging, get_logger
from .v2_streaming import iter_sensor_events, simulate_factory_stream


logger = get_logger(__name__)


def _load_mqtt_client():
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError(
            "paho-mqtt is required for MQTT replay. Install requirements.txt or run through Docker Compose."
        ) from exc
    return mqtt.Client()


def publish_simulated_events(
    broker_host: str = MQTT_BROKER_HOST,
    broker_port: int = MQTT_BROKER_PORT,
    topic: str = MQTT_SENSOR_TOPIC,
    machines: int = 4,
    steps: int = 60,
    delay_seconds: float = 0.0,
    seed: int = RANDOM_STATE + 101,
) -> int:
    client = _load_mqtt_client()
    client.connect(broker_host, broker_port, keepalive=30)
    client.loop_start()

    stream_df = simulate_factory_stream(num_machines=machines, steps=steps, seed=seed)
    count = 0
    for event in iter_sensor_events(stream_df):
        payload = json.dumps(event.__dict__)
        client.publish(topic, payload=payload, qos=0)
        count += 1
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    client.loop_stop()
    client.disconnect()
    logger.info("mqtt_replay_completed", extra={"events": count, "topic": topic})
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay simulated factory sensor events to MQTT.")
    parser.add_argument("--broker-host", default=MQTT_BROKER_HOST)
    parser.add_argument("--broker-port", type=int, default=MQTT_BROKER_PORT)
    parser.add_argument("--topic", default=MQTT_SENSOR_TOPIC)
    parser.add_argument("--machines", type=int, default=4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    events = publish_simulated_events(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        topic=args.topic,
        machines=args.machines,
        steps=args.steps,
        delay_seconds=args.delay_seconds,
    )
    print(json.dumps({"published_events": events, "topic": args.topic}, indent=2))


if __name__ == "__main__":
    main()
