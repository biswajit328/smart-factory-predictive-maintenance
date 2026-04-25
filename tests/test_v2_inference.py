import unittest

import numpy as np

from src.config import FEATURE_GROUPS, SIMULATED_SENSOR_COLUMNS
from src.v2_inference import NeuralPredictiveMaintenanceService


class IdentityScaler:
    def transform(self, values):
        return values


class FakeModel:
    def predict(self, inputs, verbose=0):
        batch_size = next(iter(inputs.values())).shape[0]
        return np.full((batch_size, 1), 0.42)


def build_test_service(strict: bool) -> NeuralPredictiveMaintenanceService:
    metadata = {
        "window_size": 1,
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
        strict=strict,
    )


def build_fused_reading():
    row = {
        "machine_id": "M001",
        "machine_type": "H",
        "timestamp": "2026-01-01T00:00:00Z",
        "step": 1,
        "breakdown_event": 0,
        "failure_next_horizon": 0,
    }
    row.update({sensor: 1.0 for sensor in SIMULATED_SENSOR_COLUMNS})
    return row


class V2InferenceCoercionTests(unittest.TestCase):
    def test_non_strict_mode_logs_warning_and_falls_back(self):
        service = build_test_service(strict=False)
        reading = build_fused_reading()
        reading["air_temp_k"] = "invalid-float"

        with self.assertLogs("src.v2_inference", level="WARNING") as logs:
            result = service.ingest_fused_reading(reading)

        self.assertIsNotNone(result)
        self.assertTrue(any("fused_reading.air_temp_k" in entry for entry in logs.output))
        self.assertTrue(any("Falling back to 0.0" in entry for entry in logs.output))

    def test_strict_mode_raises_on_invalid_value(self):
        service = build_test_service(strict=True)
        reading = build_fused_reading()
        reading["air_temp_k"] = "invalid-float"

        with self.assertRaisesRegex(ValueError, "fused_reading.air_temp_k"):
            service.ingest_fused_reading(reading)


if __name__ == "__main__":
    unittest.main()
