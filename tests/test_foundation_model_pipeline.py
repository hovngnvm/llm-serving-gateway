"""
Automated Test Suite for Foundation Model Adaptation & Evaluation Pipeline.
Tests configuration schema, dataset validation & PII audit, SFT/QLoRA dry-run/smoke-test, evaluation, and manifest contract.
"""

from pathlib import Path
import unittest

from training.src.config_schema import load_pipeline_config
from training.src.dataset_validator import DatasetValidator
from training.src.train_engine import TrainEngine
from training.src.eval_engine import EvalEngine
from training.src.manifest_builder import ManifestBuilder
from training.src.utils.paths import PROJECT_ROOT
from training.run_pipeline import run_pipeline


class TestFoundationModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dev_config_path = str(PROJECT_ROOT / "training" / "configs" / "dev.yaml")
        cls.config = load_pipeline_config(cls.dev_config_path)

    def test_config_schema_validation(self) -> None:
        """Validates YAML configuration integrity and defaults."""
        config = self.config
        self.assertEqual(config.model.base_model_name, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(config.qlora.r, 16)
        self.assertEqual(config.qlora.lora_alpha, 32)
        self.assertEqual(config.intent_routing.adapter_name, "financial_adapter")
        self.assertIn("chuyển tiền", config.intent_routing.keywords)

    def test_dataset_validation_and_pii_audit(self) -> None:
        """Validates ChatML canonicalization, dedup, PII scanning, and splits."""
        validator = DatasetValidator(self.config.dataset, seed=self.config.seed)
        report = validator.process()

        self.assertEqual(report["status"], "success")
        self.assertGreater(report["train_sample_count"], 0)
        self.assertGreater(report["val_sample_count"], 0)
        self.assertTrue((PROJECT_ROOT / report["artifacts"]["train_file"]).exists())
        self.assertTrue((PROJECT_ROOT / report["artifacts"]["val_file"]).exists())

        pii_counts = report["pii_audit_counts"]
        self.assertIsInstance(pii_counts, dict)
        self.assertIn("CITIZEN_ID", pii_counts)
        self.assertIn("PHONE_NUMBER", pii_counts)

    def test_train_engine_dry_run(self) -> None:
        """Validates mathematical graph verification and VRAM estimation."""
        train_engine = TrainEngine(self.config)
        train_path = str(PROJECT_ROOT / "training" / "data" / "processed" / "train.jsonl")
        dry_run = train_engine.execute_dry_run(train_path)

        self.assertEqual(dry_run["status"], "passed")
        self.assertGreater(dry_run["trainable_parameters"], 0)
        self.assertLess(dry_run["trainable_percentage"], 5.0)
        self.assertLess(dry_run["estimated_vram_gb"], 6.0)
        self.assertIn("6GB VRAM", dry_run["hardware_compatibility"])

    def test_train_engine_smoke_test(self) -> None:
        """Validates gradient flow and adapter artifact generation."""
        train_engine = TrainEngine(self.config)
        smoke_res = train_engine.execute_contract_verification()

        self.assertEqual(smoke_res["status"], "success")
        self.assertLess(smoke_res["final_loss"], smoke_res["initial_loss"])

        adapter_dir = PROJECT_ROOT / smoke_res["adapter_dir"]
        self.assertTrue((adapter_dir / "adapter_config.json").exists())
        self.assertTrue((adapter_dir / "adapter_model.bin").exists())

    def test_eval_engine_metrics(self) -> None:
        """Validates model evaluation comparing Base vs LoRA."""
        eval_engine = EvalEngine(self.config)
        val_path = str(PROJECT_ROOT / "training" / "data" / "processed" / "val.jsonl")
        eval_res = eval_engine.evaluate(val_path)

        self.assertEqual(eval_res["status"], "success")
        benchmarks = eval_res["benchmark_results"]

        base_validity = benchmarks["base_zero_shot"]["json_validity_rate"]
        lora_validity = benchmarks["lora_adapter"]["json_validity_rate"]
        self.assertGreater(lora_validity, 0.0)
        self.assertGreaterEqual(base_validity, 0.0)

        lora_lat = benchmarks["lora_adapter"]["avg_latency_ms"]
        self.assertGreater(lora_lat, 0.0)

    def test_manifest_builder_contract(self) -> None:
        """Validates MLOps reproducible manifest.json contract."""
        manifest_builder = ManifestBuilder(self.config)
        manifest = manifest_builder.build_manifest()

        self.assertEqual(manifest["manifest_version"], "1.0.0")
        self.assertEqual(manifest["pipeline_name"], self.config.pipeline_name)
        self.assertEqual(manifest["intent_routing"]["adapter_name"], "financial_adapter")
        self.assertTrue(manifest["deployment_readiness"]["production_grade"])
        self.assertTrue(manifest["deployment_readiness"]["vllm_dynamic_lora_ready"])

    def test_cli_orchestrator_smoke_test(self) -> None:
        """Validates full CLI orchestrator execution."""
        run_pipeline(self.dev_config_path, mode="smoke-test", target_stage=0)


if __name__ == "__main__":
    unittest.main()
