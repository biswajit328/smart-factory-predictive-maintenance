import unittest

import numpy as np

from src.model import choose_probability_threshold


class ThresholdSelectionTests(unittest.TestCase):
    def test_threshold_selection_returns_valid_range(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.05, 0.10, 0.40, 0.62, 0.81, 0.93])

        result = choose_probability_threshold(y_true, probabilities, precision_floor=0.5)

        self.assertGreaterEqual(result.threshold, 0.0)
        self.assertLessEqual(result.threshold, 1.0)
        self.assertGreaterEqual(result.precision, 0.0)
        self.assertGreaterEqual(result.recall, 0.0)


if __name__ == "__main__":
    unittest.main()
