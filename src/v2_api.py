from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from .v2_inference import NeuralPredictiveMaintenanceService, run_live_demo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .config import (
    SIMULATED_SENSOR_COLUMNS,
    V2_DASHBOARD_PATH,
    V2_LIVE_PREDICTIONS_PATH,
    V2_METADATA_PATH,
    V2_MODEL_PATH,
    V2_SCALERS_PATH,
)


class FusedReadingRequest(BaseModel):
    machine_id: str = Field(..., description="Machine identifier such as M000.")
    machine_type: Literal["H", "L", "M"]
    timestamp: str
    step: int = 0
    air_temp_k: float
    process_temp_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    vibration_mm_s: float
    pressure_bar: float
    current_a: float
    acoustic_db: float
    humidity_pct: float
    breakdown_event: int = 0
    failure_next_horizon: int = 0


class SensorEventRequest(BaseModel):
    machine_id: str
    machine_type: Literal["H", "L", "M"]
    timestamp: str
    step: int = 0
    sensor_name: Literal[
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "vibration_mm_s",
        "pressure_bar",
        "current_a",
        "acoustic_db",
        "humidity_pct",
    ]
    sensor_value: float
    breakdown_event: int = 0
    failure_next_horizon: int = 0


class SensorEventBatchRequest(BaseModel):
    events: list[SensorEventRequest]


class SimulationRequest(BaseModel):
    output_path: str = str(V2_LIVE_PREDICTIONS_PATH)
    dashboard_path: str = str(V2_DASHBOARD_PATH)


class ApiRuntime:
    def __init__(self, service: NeuralPredictiveMaintenanceService | None = None):
        self.service = service
        self.loaded_at: str | None = None

    def artifact_status(self) -> dict[str, bool]:
        return {
            "model_exists": Path(V2_MODEL_PATH).exists(),
            "scalers_exists": Path(V2_SCALERS_PATH).exists(),
            "metadata_exists": Path(V2_METADATA_PATH).exists(),
        }

    def ensure_service(self) -> NeuralPredictiveMaintenanceService:
        if self.service is None:
            status = self.artifact_status()
            if not all(status.values()):
                missing = [name for name, exists in status.items() if not exists]
                raise FileNotFoundError(
                    "Missing v2 artifacts: " + ", ".join(missing) + ". Run `python -m src.v2_train` first."
                )
            self.service = NeuralPredictiveMaintenanceService(
                model_path=V2_MODEL_PATH,
                scalers_path=V2_SCALERS_PATH,
                metadata_path=V2_METADATA_PATH,
            )
            self.loaded_at = datetime.utcnow().isoformat() + "Z"
        return self.service


def create_app(service: NeuralPredictiveMaintenanceService | None = None) -> FastAPI:
    runtime = ApiRuntime(service=service)
    app = FastAPI(
        title="Smart Factory Predictive Maintenance API",
        version="0.1.0",
        description="FastAPI service layer for the neural predictive maintenance v2 model.",
    )
    app.state.runtime = runtime

    @app.get("/health")
    def health() -> dict[str, object]:
        status = runtime.artifact_status()
        return {
            "status": "ok" if all(status.values()) else "artifacts_missing",
            "artifacts": status,
            "service_loaded": runtime.service is not None,
            "loaded_at": runtime.loaded_at,
        }

    @app.get("/metadata")
    def metadata() -> dict[str, object]:
        try:
            service_instance = runtime.ensure_service()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return service_instance.metadata

    @app.post("/stream/reset")
    def reset_stream() -> dict[str, str]:
        try:
            service_instance = runtime.ensure_service()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        service_instance.reset_state()
        return {"status": "reset", "message": "Stream buffers cleared."}

    @app.post("/predict/fused")
    def predict_fused(reading: FusedReadingRequest) -> dict[str, object]:
        try:
            service_instance = runtime.ensure_service()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        payload = reading.model_dump()
        prediction = service_instance.ingest_fused_reading(payload)
        if prediction is None:
            buffer_size = len(service_instance.machine_buffers[payload["machine_id"]])
            return {
                "status": "warming_up",
                "message": "The model needs more readings for this machine before it can score risk.",
                "current_window_size": buffer_size,
                "required_window_size": service_instance.window_size,
            }
        return {"status": "prediction", "result": prediction}

    @app.post("/predict/events")
    def predict_events(event_batch: SensorEventBatchRequest) -> dict[str, object]:
        try:
            service_instance = runtime.ensure_service()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        responses = []
        for event in event_batch.events:
            payload = event.model_dump()
            payload["sensor_group"] = "unknown"
            result = service_instance.ingest_event(payload)
            responses.append(
                {
                    "event": {
                        "machine_id": payload["machine_id"],
                        "timestamp": payload["timestamp"],
                        "sensor_name": payload["sensor_name"],
                    },
                    "status": "prediction" if result is not None else "buffering",
                    "result": result,
                }
            )
        return {"status": "ok", "responses": responses}

    @app.post("/simulate/run")
    def simulate(request: SimulationRequest) -> dict[str, object]:
        try:
            predictions_df, dashboard_path = run_live_demo(
                output_path=request.output_path,
                dashboard_path=request.dashboard_path,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        top_alerts = (
            predictions_df.sort_values(
                ["failure_probability", "timestamp"],
                ascending=[False, True],
            )
            .head(5)
            .to_dict(orient="records")
        )
        return {
            "status": "ok",
            "prediction_rows": int(len(predictions_df)),
            "dashboard_path": str(dashboard_path),
            "output_path": str(request.output_path),
            "top_alerts": top_alerts,
        }

    @app.get("/examples/fused-reading")
    def example_fused_reading() -> dict[str, object]:
        example = {
            "machine_id": "M000",
            "machine_type": "H",
            "timestamp": "2026-01-01T08:00:00",
            "step": 0,
            "air_temp_k": 299.8,
            "process_temp_k": 311.2,
            "rotational_speed_rpm": 1405.0,
            "torque_nm": 49.2,
            "tool_wear_min": 22.0,
            "vibration_mm_s": 2.1,
            "pressure_bar": 8.7,
            "current_a": 44.8,
            "acoustic_db": 68.2,
            "humidity_pct": 47.5,
            "breakdown_event": 0,
            "failure_next_horizon": 0,
        }
        return {"required_sensor_columns": SIMULATED_SENSOR_COLUMNS, "example": example}

    return app


app = create_app()


def main() -> None:
    uvicorn.run("src.v2_api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

