"""
Dataset Validation, Normalization & Security Scanner.
Converts raw inputs to canonical ChatML, deduplicates, scans PII, and splits data.
"""

import json
import hashlib
import random
import re
from pathlib import Path
from typing import Any
from training.src.config_schema import DatasetConfig, PROJECT_ROOT


class DatasetValidator:
    def __init__(self, config: DatasetConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        random.seed(seed)

        # Decree 13 PII patterns for dataset compliance checking
        self.pii_patterns = {
            "CITIZEN_ID": re.compile(r"\b\d{12}\b"),
            "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
            "BANK_ACCOUNT": re.compile(r"(?i)\b(?:stk|tài khoản|tk ngân hàng|số tk)[:\s]*([0-9]{8,16})\b"),
            "PHONE_NUMBER": re.compile(r"(?:\+84|0)(?:3[2-9]|5[689]|7[06-9]|8[1-9]|9[0-9])\d{7}\b"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "TAX_ID": re.compile(r"\b\d{10}(?:-\d{3})?\b"),
            "PASSPORT_VN": re.compile(r"\b[BCDEFGHMP]\d{7,8}\b"),
            "IP_ADDRESS": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        }

    def canonicalize_to_chatml(self, item: dict[str, Any]) -> dict[str, Any]:
        """Converts an Alpaca or ShareGPT item into a canonical ChatML format."""
        if "messages" in item and isinstance(item["messages"], list):
            return item

        if "instruction" in item or "input" in item:
            system_prompt = item.get("instruction", "You are a financial information extraction assistant.")
            user_content = item.get("input", "")
            assistant_content = item.get("output", "")

            if isinstance(assistant_content, dict):
                assistant_content = json.dumps(assistant_content, ensure_ascii=False)

            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": str(assistant_content)}
                ]
            }

        if "conversations" in item and isinstance(item["conversations"], list):
            messages = []
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            for turn in item["conversations"]:
                mapped_role = role_map.get(turn.get("from", "user"), "user")
                messages.append({"role": mapped_role, "content": turn.get("value", "")})
            return {"messages": messages}

        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(item)},
                {"role": "assistant", "content": ""}
            ]
        }

    def compute_sha256(self, file_path: str) -> str:
        """Computes SHA-256 hash of a file for lineage tracking."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def scan_pii(self, text: str) -> dict[str, int]:
        """Scans text against Decree 13 PII patterns."""
        pii_counts: dict[str, int] = {}
        for entity, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                pii_counts[entity] = len(matches)
        return pii_counts

    def process(self) -> dict[str, Any]:
        """Executes full dataset normalization, deduplication, PII scanning, and train/val/calibration splitting."""
        raw_path = Path(self.config.raw_data_path)
        if not raw_path.is_absolute() and not raw_path.exists():
            fallback_raw = PROJECT_ROOT / self.config.raw_data_path
            if fallback_raw.exists():
                raw_path = fallback_raw

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

        raw_sha256 = self.compute_sha256(str(raw_path))

        raw_records = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    raw_records.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON decode error at line {line_idx} in {raw_path}: {e}")

        if not raw_records:
            raise ValueError(f"Dataset at {raw_path} is empty.")

        seen_prompts = set()
        deduped_records = []
        total_pii_stats: dict[str, int] = {}

        for record in raw_records:
            chatml = self.canonicalize_to_chatml(record)
            user_msg = next((m["content"] for m in chatml["messages"] if m["role"] == "user"), "")
            prompt_hash = hashlib.md5(user_msg.encode("utf-8")).hexdigest()

            if prompt_hash in seen_prompts:
                continue
            seen_prompts.add(prompt_hash)
            deduped_records.append(chatml)

            full_text = " ".join(m["content"] for m in chatml["messages"])
            for entity, count in self.scan_pii(full_text).items():
                total_pii_stats[entity] = total_pii_stats.get(entity, 0) + count

        shuffled = list(deduped_records)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(n_total * self.config.train_split_ratio))
        train_records = shuffled[:n_train]
        val_records = shuffled[n_train:] if n_train < n_total else shuffled[:max(1, int(n_total * 0.1))]

        calib_count = min(self.config.calibration_samples, len(train_records))
        calibration_records = train_records[:calib_count]
        if len(calibration_records) < self.config.calibration_samples:
            multiplier = (self.config.calibration_samples // len(train_records)) + 1
            calibration_records = (train_records * multiplier)[:self.config.calibration_samples]

        out_dir = Path(self.config.processed_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / self.config.processed_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        train_file = out_dir / "train.jsonl"
        val_file = out_dir / "val.jsonl"
        calib_file = out_dir / "calibration.jsonl"
        report_file = out_dir / "validation_report.json"

        with open(train_file, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for r in val_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(calib_file, "w", encoding="utf-8") as f:
            for r in calibration_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def _to_portable(p: Path) -> str:
            try:
                return p.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return p.as_posix()

        report = {
            "status": "success",
            "raw_dataset_path": _to_portable(raw_path),
            "raw_dataset_sha256": raw_sha256,
            "raw_sample_count": len(raw_records),
            "deduped_sample_count": len(deduped_records),
            "train_sample_count": len(train_records),
            "val_sample_count": len(val_records),
            "calibration_sample_count": len(calibration_records),
            "pii_audit_counts": total_pii_stats,
            "artifacts": {
                "train_file": _to_portable(train_file),
                "val_file": _to_portable(val_file),
                "calibration_file": _to_portable(calib_file),
            }
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

