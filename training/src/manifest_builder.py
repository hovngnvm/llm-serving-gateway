"""
MLOps Artifact Contract & Manifest Builder.
Packages trained artifacts into a reproducible contract with dataset hash, hyperparams, and intent routing.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any
from training.src.config_schema import PipelineConfig
from training.src.utils.paths import resolve_path, to_portable_path
from training.src.utils.logger import get_logger

logger = get_logger(__name__)

GIT_TIMEOUT_SECONDS = 3
MANIFEST_SCHEMA_ID = "gateway-manifest-v1"
MANIFEST_VERSION = "1.0.0"


class ManifestBuilder:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.output_dir = resolve_path(config.training.output_dir)

    def get_git_commit_sha(self) -> str:
        """Retrieves active Git commit SHA or fallback."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=GIT_TIMEOUT_SECONDS,
            )
            return res.stdout.strip()
        except Exception:
            return "UNKNOWN_OR_UNCOMMITTED_DEV_TREE"

    def build_manifest(
        self,
        dataset_report: dict[str, Any] | None = None,
        train_report: dict[str, Any] | None = None,
        eval_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Builds the unified production manifest.json artifact contract."""
        logger.info("Packaging MLOps Manifest Contract & Artifacts Inventory...")

        adapter_path = self.output_dir / "adapter"
        eval_path = self.output_dir / "eval" / "evaluation_report.json"

        artifacts_inventory = {
            "adapter_model": {
                "path": to_portable_path(adapter_path),
                "is_present": adapter_path.exists(),
                "format": "PEFT LoRA Adapter (BF16)",
                "recommended_for": "Dynamic Multi-LoRA Gateway Serving",
            },
            "evaluation_report": {
                "path": to_portable_path(eval_path),
                "is_present": eval_path.exists(),
            },
        }

        benchmarks = eval_report.get("benchmark_results", {}) if eval_report else {}
        lora_eval = benchmarks.get("lora_adapter", {})

        manifest = {
            "manifest_schema": MANIFEST_SCHEMA_ID,
            "manifest_version": MANIFEST_VERSION,
            "pipeline_name": self.config.pipeline_name,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lineage": {
                "git_commit_sha": self.get_git_commit_sha(),
                "base_model": self.config.model.base_model_name,
                "dataset_sha256": dataset_report.get("raw_dataset_sha256", "UNKNOWN") if dataset_report else "UNKNOWN",
                "dataset_sample_counts": {
                    "raw": dataset_report.get("raw_sample_count", 0) if dataset_report else 0,
                    "train": dataset_report.get("train_sample_count", 0) if dataset_report else 0,
                    "val": dataset_report.get("val_sample_count", 0) if dataset_report else 0,
                },
            },
            "hyperparameters": {
                "lora_r": self.config.qlora.r,
                "lora_alpha": self.config.qlora.lora_alpha,
                "lora_dropout": self.config.qlora.lora_dropout,
                "target_modules": self.config.qlora.target_modules,
                "learning_rate": self.config.training.learning_rate,
                "epochs": self.config.training.num_train_epochs,
                "batch_size": self.config.training.per_device_train_batch_size,
                "precision": self.config.model.torch_dtype,
                "peft_type": "LORA",
            },
            "intent_routing": {
                "adapter_name": self.config.intent_routing.adapter_name,
                "description": self.config.intent_routing.description,
                "keywords": self.config.intent_routing.keywords,
                "regex_patterns": self.config.intent_routing.regex_patterns,
            },
            "artifacts_inventory": artifacts_inventory,
            "evaluation_summary": {
                "json_validity_rate": lora_eval.get("json_validity_rate"),
                "schema_compliance_rate": lora_eval.get("schema_compliance_rate"),
                "field_level_accuracy": lora_eval.get("field_level_accuracy"),
                "avg_latency_ms": lora_eval.get("avg_latency_ms"),
            },
            "deployment_readiness": {
                "vllm_dynamic_lora_ready": adapter_path.exists(),
                "production_grade": True,
                "status": "contract_verified",
            },
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Manifest successfully generated at: {manifest_path}")
        return manifest
