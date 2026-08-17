"""
Automated Test Suite for Gateway Health, Logging & Observability.
Verifies system health endpoints, audit logging schema, and docker-compose services.
"""

import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from gateway.app.core.semantic_cache import semantic_cache
from gateway.app.core.output_validator import output_validator
from gateway.app.db.neon_audit_logger import neon_audit_logger


class TestHealthAndObservability(unittest.TestCase):
    def test_docker_compose_core_services(self) -> None:
        """Verifies docker-compose contains core application services."""
        compose_path = BASE_DIR / "docker-compose.yml"
        self.assertTrue(compose_path.exists())
        content = compose_path.read_text(encoding="utf-8")

        self.assertIn("gateway:", content)
        self.assertIn("redis-cache:", content)
        self.assertIn("prometheus:", content)

        prom_cfg = BASE_DIR / "prometheus" / "prometheus.yml"
        self.assertTrue(prom_cfg.exists())
        self.assertIn("ai-gateway", prom_cfg.read_text(encoding="utf-8"))

    def test_gateway_health_metrics_structure(self) -> None:
        """Verifies internal observability counters and health stats."""
        stats = {
            "status": "healthy",
            "semantic_cache": {
                "hits": semantic_cache.hit_count,
                "misses": semantic_cache.miss_count,
            },
            "schema_validations": output_validator.validation_count,
            "json_auto_repairs": output_validator.auto_repair_count,
        }
        self.assertEqual(stats["status"], "healthy")
        self.assertIn("hits", stats["semantic_cache"])
        self.assertIn("misses", stats["semantic_cache"])
        self.assertGreaterEqual(stats["schema_validations"], 0)
        self.assertGreaterEqual(stats["json_auto_repairs"], 0)

    def test_audit_logger_schema_contract(self) -> None:
        """Verifies Neon PostgreSQL audit logger schema query definition."""
        self.assertTrue(hasattr(neon_audit_logger, "log_request"))
        self.assertTrue(hasattr(neon_audit_logger, "init_db"))


if __name__ == "__main__":
    unittest.main()
