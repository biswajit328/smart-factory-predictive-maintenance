import unittest

from fastapi.testclient import TestClient

from src.v2_api import create_app
from src.v2_inference import NeuralPredictiveMaintenanceService
from src.v2_streaming import simulate_factory_stream
from src.v2_train import train_smart_factory_v2


class V2ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle, _ = train_smart_factory_v2(
            num_machines=6,
            steps=70,
            epochs=2,
            batch_size=16,
            save_artifacts=False,
        )
        service = NeuralPredictiveMaintenanceService(
            model=bundle["model"],
            scalers=bundle["scalers"],
            metadata=bundle["metadata"],
        )
        cls.client = TestClient(create_app(service=service))
        cls.sample_stream = simulate_factory_stream(num_machines=3, steps=30, seed=77)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("status", payload)

    def test_fused_prediction_endpoint_warms_up_then_scores(self):
        machine_rows = self.sample_stream[self.sample_stream["machine_id"] == "M000"].head(20)
        last_payload = None
        for row in machine_rows.to_dict(orient="records"):
            last_payload = {
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
            response = self.client.post("/predict/fused", json=last_payload)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "prediction")
        self.assertIn("failure_probability", payload["result"])

    def test_stream_reset_endpoint(self):
        response = self.client.post("/stream/reset")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "reset")


if __name__ == "__main__":
    unittest.main()
