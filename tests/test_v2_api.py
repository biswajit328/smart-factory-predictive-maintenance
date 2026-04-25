import unittest

import numpy as np
from fastapi.testclient import TestClient

from src.config import FEATURE_GROUPS, SIMULATED_SENSOR_COLUMNS
from src.v2_api import create_app
from src.v2_inference import NeuralPredictiveMaintenanceService
from src.v2_streaming import simulate_factory_stream


class IdentityScaler:
    def transform(self, values):
        return values


class FakeModel:
    def predict(self, inputs, verbose=0):
        batch_size = next(iter(inputs.values())).shape[0]
        return np.full((batch_size, 1), 0.72)


def build_test_service() -> NeuralPredictiveMaintenanceService:
    metadata = {
        "window_size": 3,
        "probability_threshold": 0.5,
        "required_sensor_columns": SIMULATED_SENSOR_COLUMNS,
        "type_values": ["H", "L", "M"],
        "feature_groups": FEATURE_GROUPS,
    }
    scalers = {group_name: IdentityScaler() for group_name in FEATURE_GROUPS}
    return NeuralPredictiveMaintenanceService(
        model=FakeModel(),
        scalers=scalers,
        metadata=metadata,
    )


class V2ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(service=build_test_service()))
        self.sample_stream = simulate_factory_stream(num_machines=2, steps=8, seed=77)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())

    def test_infrastructure_endpoint(self):
        response = self.client.get("/infrastructure")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("redis", payload)
        self.assertIn("postgres", payload)
        self.assertIn("mqtt", payload)

    def test_fused_prediction_endpoint_warms_up_then_scores(self):
        machine_rows = self.sample_stream[self.sample_stream["machine_id"] == "M000"].head(3)

        for row in machine_rows.to_dict(orient="records"):
            payload = {
                "machine_id": row["machine_id"],
                "machine_type": row["machine_type"],
                "timestamp": row["timestamp"],
                "step": int(row["step"]),
                "air_temp_k": row["air_temp_k"],
                "process_temp_k": row["process_temp_k"],
                "rotational_speed_rpm": row["rotational_speed_rpm"],
                "torque_nm": row["torque_nm"],
                "tool_wear_min": row["tool_wear_min"],
                "vibration_mm_s": row["vibration_mm_s"],
                "pressure_bar": row["pressure_bar"],
                "current_a": row["current_a"],
                "acoustic_db": row["acoustic_db"],
                "humidity_pct": row["humidity_pct"],
                "breakdown_event": int(row["breakdown_event"]),
                "failure_next_horizon": int(row["failure_next_horizon"]),
            }
            response = self.client.post("/predict/fused", json=payload)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "prediction")
        self.assertEqual(payload["result"]["failure_probability"], 0.72)
        self.assertTrue(payload["result"]["classification_flag"])

    def test_stream_reset_endpoint(self):
        response = self.client.post("/stream/reset")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "reset")


if __name__ == "__main__":
    unittest.main()
