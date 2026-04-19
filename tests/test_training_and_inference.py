import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.config import RAW_INPUT_COLUMNS
from src.inference import PredictiveMaintenanceService
from src.preprocess import load_raw_data
from src.train import train_project_model


class TrainingAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_df = load_raw_data()
        positive_rows = raw_df[raw_df["Machine failure"] == 1].sample(
            n=80,
            random_state=42,
        )
        negative_rows = raw_df[raw_df["Machine failure"] == 0].sample(
            n=320,
            random_state=42,
        )
        sample_df = pd.concat([positive_rows, negative_rows], ignore_index=True)

        cls.bundle, cls.metrics = train_project_model(
            raw_df=sample_df,
            save_artifacts=False,
            model_params={"n_estimators": 60},
        )
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.bundle_path = Path(cls.temp_dir.name) / "bundle.joblib"
        joblib.dump(cls.bundle, cls.bundle_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_training_smoke_metrics(self):
        self.assertIn("classification", self.metrics)
        self.assertGreater(self.metrics["classification"]["roc_auc"], 0.5)
        self.assertIn("threshold_selection", self.metrics)

    def test_service_predict_one_returns_expected_keys(self):
        service = PredictiveMaintenanceService(self.bundle_path)
        sample_record = load_raw_data().iloc[0][RAW_INPUT_COLUMNS].to_dict()

        result = service.predict_one(sample_record)

        self.assertIn("failure_probability", result)
        self.assertIn("classification_flag", result)
        self.assertIn("anomaly_flag", result)
        self.assertIn("recommended_action", result)


if __name__ == "__main__":
    unittest.main()

