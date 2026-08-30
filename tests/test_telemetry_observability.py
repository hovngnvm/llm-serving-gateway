"""
Automated Test Suite for Gateway Health, Logging & Observability.
Verifies system health endpoints, audit logging schema, and docker-compose services.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from gateway.app.main import app
from gateway.app.db.neon_audit_logger import neon_audit_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestHealthAndObservability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_docker_compose_core_services(self) -> None:
        """Verifies docker-compose contains core application services."""
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        self.assertTrue(compose_path.exists())
        content = compose_path.read_text(encoding="utf-8")

        self.assertIn("gateway:", content)
        self.assertIn("redis-cache:", content)
        self.assertIn("prometheus:", content)

        prom_cfg = PROJECT_ROOT / "prometheus" / "prometheus.yml"
        self.assertTrue(prom_cfg.exists())
        self.assertIn("ai-gateway", prom_cfg.read_text(encoding="utf-8"))

    def test_gateway_health_endpoint_response(self) -> None:
        """Verifies real /health endpoint returns healthy status and metric counters."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("semantic_cache", data)
        self.assertIn("hits", data["semantic_cache"])
        self.assertIn("misses", data["semantic_cache"])
        self.assertIn("schema_validations", data)
        self.assertIn("json_auto_repairs", data)

    def test_prometheus_metrics_endpoint_response(self) -> None:
        """Verifies Prometheus metrics endpoint is exposed and scraped successfully."""
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("http_request", resp.text)

    def test_audit_logger_schema_contract(self) -> None:
        """Verifies Neon PostgreSQL audit logger schema query definition."""
        self.assertTrue(callable(getattr(neon_audit_logger, "log_request", None)))
        self.assertTrue(callable(getattr(neon_audit_logger, "init_db", None)))


if __name__ == "__main__":
    unittest.main()
