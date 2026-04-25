import unittest

from fastapi.testclient import TestClient

from src.v2_dashboard_server import create_app


class V2DashboardServerTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("dashboard_exists", response.json())

    def test_index_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Smart Factory Predictive Maintenance Demo", response.text)


if __name__ == "__main__":
    unittest.main()
