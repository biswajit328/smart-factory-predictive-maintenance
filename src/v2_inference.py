from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd

from .config import (
    FEATURE_GROUPS,
    RANDOM_STATE,
    SIMULATED_SENSOR_COLUMNS,
    V2_DASHBOARD_PATH,
    V2_LIVE_PREDICTIONS_PATH,
    V2_METADATA_PATH,
    V2_MODEL_PATH,
    V2_OUTPUT_DIR,
    V2_SCALERS_PATH,
)
from .v2_dashboard import build_dashboard_report
from .v2_neural import transform_single_window
from .v2_streaming import iter_sensor_events, simulate_factory_stream


class SensorEventFusionBuffer:
    def __init__(self, required_sensors: list[str] | None = None):
        self.required_sensors = set(required_sensors or SIMULATED_SENSOR_COLUMNS)
        self.partial_rows: dict[tuple[str, str], dict[str, object]] = {}

    def ingest_event(self, event: dict[str, object]) -> dict[str, object] | None:
        key = (str(event["machine_id"]), str(event["timestamp"]))
        row = self.partial_rows.setdefault(
            key,
            {
                "machine_id": str(event["machine_id"]),
                "machine_type": str(event["machine_type"]),
                "timestamp": str(event["timestamp"]),
                "step": int(event.get("step", 0)),
                "breakdown_event": int(event.get("breakdown_event", 0)),
                "failure_next_horizon": int(event.get("failure_next_horizon", 0)),
            },
        )
        row[str(event["sensor_name"])] = float(event["sensor_value"])

        if self.required_sensors.issubset(row.keys()):
            fused_row = row.copy()
            del self.partial_rows[key]
            return fused_row
        return None


class NeuralPredictiveMaintenanceService:
    def __init__(
        self,
        model=None,
        scalers=None,
        metadata: dict[str, object] | None = None,
        model_path: str | Path = V2_MODEL_PATH,
        scalers_path: str | Path = V2_SCALERS_PATH,
        metadata_path: str | Path = V2_METADATA_PATH,
    ):
        self.metadata = metadata or json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.model = model if model is not None else keras.models.load_model(model_path)
        self.scalers = scalers if scalers is not None else joblib.load(scalers_path)
        self.window_size = int(self.metadata["window_size"])
        self.probability_threshold = float(self.metadata["probability_threshold"])
        self.machine_buffers = defaultdict(lambda: deque(maxlen=self.window_size))
        self.event_fusion_buffer = SensorEventFusionBuffer(
            required_sensors=self.metadata["required_sensor_columns"]
        )

    def _build_feature_row(self, fused_reading: dict[str, object]) -> dict[str, float]:
        row = {column: float(fused_reading[column]) for column in SIMULATED_SENSOR_COLUMNS}
        machine_type = str(fused_reading["machine_type"])
        for type_value in self.metadata["type_values"]:
            row[f"type_{type_value}"] = 1.0 if machine_type == type_value else 0.0
        return row

    def _window_to_model_inputs(self, machine_id: str) -> dict[str, np.ndarray]:
        recent_rows = list(self.machine_buffers[machine_id])
        window_frame = pd.DataFrame(recent_rows)
        grouped = {
            group_name: window_frame[columns].to_numpy(dtype=np.float32)
            for group_name, columns in FEATURE_GROUPS.items()
        }
        transformed = transform_single_window(grouped, self.scalers)
        return {
            group_name: values[np.newaxis, ...]
            for group_name, values in transformed.items()
        }

    def _build_recommendation(self, probability: float) -> tuple[str, str]:
        if probability >= self.probability_threshold:
            return "high", "Inspect this machine immediately and schedule maintenance."
        if probability >= self.probability_threshold * 0.60:
            return "medium", "Monitor this machine closely and prepare a maintenance slot."
        return "low", "Keep the machine on the normal maintenance schedule."

    def ingest_fused_reading(self, fused_reading: dict[str, object]) -> dict[str, object] | None:
        machine_id = str(fused_reading["machine_id"])
        self.machine_buffers[machine_id].append(self._build_feature_row(fused_reading))
        if len(self.machine_buffers[machine_id]) < self.window_size:
            return None

        model_inputs = self._window_to_model_inputs(machine_id)
        probability = float(self.model.predict(model_inputs, verbose=0).ravel()[0])
        risk_band, recommendation = self._build_recommendation(probability)

        return {
            "machine_id": machine_id,
            "machine_type": str(fused_reading["machine_type"]),
            "timestamp": str(fused_reading["timestamp"]),
            "step": int(fused_reading["step"]),
            "failure_probability": round(probability, 4),
            "classification_flag": bool(probability >= self.probability_threshold),
            "risk_band": risk_band,
            "maintenance_priority": round(probability * 100, 2),
            "recommended_action": recommendation,
            "breakdown_event": int(fused_reading.get("breakdown_event", 0)),
            "failure_next_horizon": int(fused_reading.get("failure_next_horizon", 0)),
        }

    def ingest_event(self, event: dict[str, object]) -> dict[str, object] | None:
        fused_reading = self.event_fusion_buffer.ingest_event(event)
        if fused_reading is None:
            return None
        return self.ingest_fused_reading(fused_reading)


def run_live_demo(
    model_path: str | Path = V2_MODEL_PATH,
    scalers_path: str | Path = V2_SCALERS_PATH,
    metadata_path: str | Path = V2_METADATA_PATH,
    output_path: str | Path = V2_LIVE_PREDICTIONS_PATH,
    dashboard_path: str | Path = V2_DASHBOARD_PATH,
) -> tuple[pd.DataFrame, Path]:
    service = NeuralPredictiveMaintenanceService(
        model_path=model_path,
        scalers_path=scalers_path,
        metadata_path=metadata_path,
    )

    stream_df = simulate_factory_stream(num_machines=6, steps=100, seed=RANDOM_STATE + 7)
    predictions = []
    for event in iter_sensor_events(stream_df):
        result = service.ingest_event(event.__dict__)
        if result is not None:
            predictions.append(result)

    predictions_df = pd.DataFrame(predictions)
    V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    predictions_df.to_csv(output_path, index=False)

    dashboard_output = build_dashboard_report(
        stream_df=stream_df,
        predictions_df=predictions_df,
        output_path=dashboard_path,
    )
    return predictions_df, dashboard_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v2 neural predictive maintenance demo.")
    parser.add_argument(
        "--live-demo",
        action="store_true",
        help="Replay a simulated sensor event stream and generate a dashboard report.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(V2_LIVE_PREDICTIONS_PATH),
        help="CSV output for live predictions.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.live_demo:
        predictions_df, dashboard_path = run_live_demo(output_path=args.output)
        top_alerts = (
            predictions_df.sort_values(
                ["failure_probability", "timestamp"],
                ascending=[False, True],
            )
            .head(5)
        )
        print(top_alerts.to_string(index=False))
        print(f"\nDashboard written to {dashboard_path}")
        return

    print("Use --live-demo after training the v2 model with 'python -m src.v2_train'.")


if __name__ == "__main__":
    main()
