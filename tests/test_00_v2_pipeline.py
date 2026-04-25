import unittest

from src.v2_inference import NeuralPredictiveMaintenanceService
from src.v2_streaming import iter_sensor_events, simulate_factory_stream
from src.v2_train import train_smart_factory_v2


class V2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.metrics = train_smart_factory_v2(
            num_machines=8,
            steps=90,
            epochs=3,
            batch_size=16,
            save_artifacts=False,
        )

    def test_neural_training_smoke_metrics(self):
        self.assertIn("classification", self.metrics)
        self.assertGreater(self.metrics["classification"]["roc_auc"], 0.6)
        self.assertIn("threshold_selection", self.metrics)
        self.assertIn("branch_importance", self.metrics)

    def test_live_service_emits_predictions(self):
        service = NeuralPredictiveMaintenanceService(
            model=self.bundle["model"],
            scalers=self.bundle["scalers"],
            metadata=self.bundle["metadata"],
        )
        stream_df = simulate_factory_stream(num_machines=3, steps=35, seed=99)

        predictions = []
        for event in iter_sensor_events(stream_df):
            result = service.ingest_event(event.__dict__)
            if result is not None:
                predictions.append(result)

        self.assertGreater(len(predictions), 0)
        self.assertIn("failure_probability", predictions[0])
        self.assertIn("risk_band", predictions[0])


if __name__ == "__main__":
    unittest.main()
