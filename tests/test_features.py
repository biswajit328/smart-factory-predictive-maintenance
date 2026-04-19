import unittest

import numpy as np
import pandas as pd

from src.features import ENGINEERED_FEATURES, prepare_model_frame


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.sample_frame = pd.DataFrame(
            [
                {
                    "UDI": 1,
                    "Product ID": "L47181",
                    "Type": "L",
                    "Air temperature [K]": 298.1,
                    "Process temperature [K]": 308.6,
                    "Rotational speed [rpm]": 1551,
                    "Torque [Nm]": 42.8,
                    "Tool wear [min]": 3,
                    "TWF": 0,
                    "HDF": 0,
                    "PWF": 0,
                    "OSF": 0,
                    "RNF": 0,
                }
            ]
        )

    def test_prepare_model_frame_creates_expected_features(self):
        prepared = prepare_model_frame(self.sample_frame)

        for feature_name in ENGINEERED_FEATURES:
            self.assertIn(feature_name, prepared.columns)
        self.assertNotIn("UDI", prepared.columns)
        self.assertNotIn("TWF", prepared.columns)

    def test_prepare_model_frame_keeps_values_finite(self):
        prepared = prepare_model_frame(self.sample_frame)
        numeric_values = prepared.select_dtypes(include=["number"]).to_numpy()
        self.assertTrue(np.isfinite(numeric_values).all())


if __name__ == "__main__":
    unittest.main()

