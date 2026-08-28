"""
Foundation Model Adaptation & Evaluation Pipeline CLI Orchestrator.
Orchestrates the 3-stage end-to-end pipeline with support for --dry-run, --smoke-test, and full --train.

Usage:
    python -m training.run_pipeline --config training/configs/dev.yaml --dry-run
    python -m training.run_pipeline --config training/configs/dev.yaml --smoke-test
    python -m training.run_pipeline --config training/configs/dev.yaml --train
    python -m training.run_pipeline --config training/configs/dev.yaml --stage 1
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.src.config_schema import load_pipeline_config
from training.src.dataset_validator import DatasetValidator
from training.src.train_engine import TrainEngine
from training.src.eval_engine import EvalEngine
from training.src.manifest_builder import ManifestBuilder
from training.src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(config_path: str, mode: str = "smoke-test", target_stage: int = 0) -> None:
    t_start = time.time()
    logger.info(f"Starting Foundation Model Pipeline (Mode: {mode.upper()}, Config: {config_path})")

    config = load_pipeline_config(config_path)
    logger.info(f"Loaded config: {config.pipeline_name} (Base Model: {config.model.base_model_name})")

    # Stage 1: Dataset Canonicalization & PII Audit
    logger.info("[Stage 1/3] Dataset Canonicalization, Dedup & PII Audit started")
    validator = DatasetValidator(config.dataset, seed=config.seed)
    dataset_report = validator.process()
    logger.info(
        f"Dataset Ready: {dataset_report['train_sample_count']} train, "
        f"{dataset_report['val_sample_count']} val"
    )
    if target_stage == 1:
        return

    train_path = dataset_report["artifacts"]["train_file"]
    val_path = dataset_report["artifacts"]["val_file"]

    # Stage 2: SFT / QLoRA Engine
    logger.info(f"[Stage 2/3] SFT / QLoRA Engine ({mode.upper()}) started")
    train_engine = TrainEngine(config)
    if mode == "dry-run":
        dry_run_report = train_engine.execute_dry_run(train_path)
        logger.info(f"Dry-Run Complete: {dry_run_report['hardware_compatibility']} (Estimated VRAM: {dry_run_report['estimated_vram_gb']} GB)")
        logger.info("Dry-run validation successful. Exiting as requested.")
        return
    elif mode == "smoke-test":
        train_report = train_engine.execute_smoke_test(train_path)
    else:
        train_report = train_engine.train(train_path, val_path)

    logger.info(f"LoRA Adapter created at: {train_report['adapter_dir']}")
    if target_stage == 2:
        return

    # Stage 3: Offline Model Evaluation & Manifest Packaging
    logger.info("[Stage 3/3] Offline Model Evaluation & Manifest Packaging started")
    eval_engine = EvalEngine(config)
    eval_report = eval_engine.evaluate(val_path, adapter_path=train_report["adapter_dir"])
    lora_res = eval_report["benchmark_results"]["lora_adapter"]
    logger.info(
        f"Evaluation Complete: LoRA JSON Validity = {lora_res['json_validity_rate']}%, "
        f"Schema Compliance = {lora_res['schema_compliance_rate']}%"
    )

    manifest_builder = ManifestBuilder(config)
    manifest = manifest_builder.build_manifest(
        dataset_report=dataset_report,
        train_report=train_report,
        eval_report=eval_report,
    )

    elapsed = round(time.time() - t_start, 2)
    logger.info(f"Pipeline execution completed successfully in {elapsed}s.")
    logger.info(f"Artifacts and manifest ready in: {config.training.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Foundation Model Adaptation & Serving Pipeline")
    parser.add_argument("--config", type=str, default="training/configs/dev.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate graph and estimate VRAM without training")
    parser.add_argument("--smoke-test", action="store_true", help="Execute end-to-end verification")
    parser.add_argument("--train", action="store_true", help="Execute full training run")
    parser.add_argument("--stage", type=int, default=0, help="Run single isolated stage (1-3)")

    args = parser.parse_args()

    mode = "smoke-test"
    if args.dry_run:
        mode = "dry-run"
    elif args.train:
        mode = "train"

    run_pipeline(config_path=args.config, mode=mode, target_stage=args.stage)


if __name__ == "__main__":
    main()
