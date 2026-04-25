import unittest

from src.config import SIMULATED_SENSOR_COLUMNS
from src.v2_streaming import simulate_factory_stream, to_sensor_events


class V2StreamingTests(unittest.TestCase):
    def test_simulator_outputs_expected_columns(self):
        stream_df = simulate_factory_stream(num_machines=6, steps=40, seed=123)

        self.assertIn("machine_id", stream_df.columns)
        self.assertIn("timestamp", stream_df.columns)
        self.assertIn("failure_next_horizon", stream_df.columns)
        self.assertTrue(set(SIMULATED_SENSOR_COLUMNS).issubset(stream_df.columns))
        self.assertGreater(stream_df["breakdown_event"].sum(), 0)

    def test_sensor_event_conversion_matches_sensor_count(self):
        stream_df = simulate_factory_stream(num_machines=4, steps=30, seed=123)
        events_df = to_sensor_events(stream_df)

        self.assertEqual(len(events_df), len(stream_df) * len(SIMULATED_SENSOR_COLUMNS))
        self.assertIn("sensor_group", events_df.columns)


if __name__ == "__main__":
    unittest.main()
