"""
Unit & Sanity test for infrastructure configuration files.
Verifies docker-compose syntax, environment variable loading, and parameters.
"""

import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class TestInfrastructureConfig(unittest.TestCase):
    def test_docker_compose_exists(self) -> None:
        compose_path = BASE_DIR / "docker-compose.yml"
        self.assertTrue(compose_path.exists(), "docker-compose.yml must exist")
        content = compose_path.read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", content)
        self.assertIn("--gpu-memory-utilization 0.40", content)
        self.assertIn("--max-model-len 2048", content)
        self.assertIn("--max-num-seqs 8", content)
        self.assertIn("redis/redis-stack:latest", content)

    def test_env_example_exists(self) -> None:
        env_path = BASE_DIR / ".env.example"
        self.assertTrue(env_path.exists(), ".env.example must exist")
        content = env_path.read_text(encoding="utf-8")
        self.assertIn("VLLM_BASE_URL", content)
        self.assertIn("REDIS_HOST", content)
        self.assertIn("NEON_DATABASE_URL", content)
        self.assertIn("HF_TOKEN", content)

    def test_requirements_file_valid(self) -> None:
        req_path = BASE_DIR / "requirements.txt"
        self.assertTrue(req_path.exists(), "requirements.txt must exist at root")
        content = req_path.read_text(encoding="utf-8")
        self.assertIn("fastapi", content)
        self.assertIn("presidio-analyzer", content)
        self.assertIn("json-repair", content)
        self.assertIn("pydantic", content)
        self.assertIn("prometheus-fastapi-instrumentator", content)


if __name__ == "__main__":
    unittest.main()
