"""
Enterprise AI Platform: Dynamic Intent Router & Zero-Touch Adapter Registry.
Automatically discovers trained LoRA adapters from MLOps manifest.json contracts
or registers new domain adapters dynamically via REST APIs without manual code changes.
"""

import re
import json
from pathlib import Path
from typing import Any
from gateway.app.config import PROJECT_ROOT, get_settings
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)


class DomainIntentRule:
    def __init__(
        self,
        target_model: str,
        description: str,
        keywords: list[str],
        regex_patterns: list[str] | None = None,
        strategy: str = "Strategy 2 (Dynamic Multi-LoRA)",
        min_priority: int = 10,
    ) -> None:
        self.target_model = target_model
        self.description = description
        self.strategy = strategy
        self.keywords = [k.lower() for k in keywords]
        self.regexes = [re.compile(p, re.IGNORECASE) for p in (regex_patterns or [])]
        self.priority = min_priority


class IntentRouter:
    def __init__(self, default_base_model: str | None = None) -> None:
        self.default_base_model = default_base_model or get_settings().vllm_model_name
        self.rules: dict[str, DomainIntentRule] = {}
        
        self.register_adapter(
            target_model="financial_adapter",
            description="Fine-Tuned LoRA Adapter for Vietnamese Financial & Transaction Extraction",
            keywords=[
                "chuyển tiền", "chuyển khoản", "stk", "số tài khoản", "tài khoản",
                "giao dịch", "vnd", "nạp tiền", "rút tiền", "sao kê", "thẻ tín dụng",
                "cccd", "cmnd", "mã pin", "otp", "ngân hàng", "vietcombank", "techcombank",
                "mbbank", "acb", "vpbank", "bidv", "vietinbank", "tpbank", "vnpay", "momo"
            ],
            regex_patterns=[
                r"\b\d{1,3}(?:[.,]\d{3})*\s*(?:vnd|đ|dong|nghìn|triệu|k)\b",
                r"(?i)\b(?:stk|tk|số thẻ)[:\s]*\d{8,19}\b"
            ],
            priority=20
        )

        self.auto_discover_from_artifacts()

    def register_adapter(
        self,
        target_model: str,
        description: str,
        keywords: list[str],
        regex_patterns: list[str] | None = None,
        strategy: str = "Strategy 2 (Dynamic Multi-LoRA)",
        priority: int = 10,
    ) -> bool:
        """Dynamically registers or updates a domain LoRA adapter rule at runtime."""
        self.rules[target_model] = DomainIntentRule(
            target_model=target_model,
            description=description,
            keywords=keywords,
            regex_patterns=regex_patterns,
            strategy=strategy,
            min_priority=priority,
        )
        logger.info(f"Registered adapter '{target_model}' dynamically ({len(keywords)} keywords).")
        return True

    def auto_discover_from_artifacts(self, root_artifacts_dir: str | Path | None = None) -> int:
        """
        Auto-scans the artifacts directory for all manifest.json files and registers
        newly trained LoRA adapters dynamically without manual code modification.
        """
        artifacts_path = Path(root_artifacts_dir) if root_artifacts_dir else (PROJECT_ROOT / "artifacts" / "runs")
        if not artifacts_path.exists():
            return 0

        discovered_count = 0
        for manifest_file in artifacts_path.glob("*/manifest.json"):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                pipeline_name = manifest.get("pipeline_name", "")
                intent_cfg = manifest.get("intent_routing")

                if intent_cfg:
                    adapter_name = intent_cfg.get("adapter_name", pipeline_name)
                    desc = intent_cfg.get("description", f"Auto-discovered adapter from {pipeline_name}")
                    keywords = intent_cfg.get("keywords", [])
                    regex_patterns = intent_cfg.get("regex_patterns", [])
                    self.register_adapter(adapter_name, desc, keywords, regex_patterns)
                    discovered_count += 1
            except Exception as e:
                logger.warning(f"Failed to auto-discover manifest {manifest_file}: {e}")

        return discovered_count

    def resolve_model(
        self,
        prompt: str,
        requested_model: str | None = None,
        pii_count: int = 0
    ) -> dict[str, Any]:
        """Dynamically resolves the target serving model or LoRA adapter."""
        prompt_clean = prompt.strip()
        prompt_lower = prompt_clean.lower()

        if requested_model and requested_model not in ("auto", "default", "", None):
            return {
                "target_model": requested_model,
                "routing_strategy": "client_explicit",
                "matched_domain": None,
                "reason": f"Client explicitly specified target model: '{requested_model}'."
            }

        sorted_rules = sorted(self.rules.values(), key=lambda r: r.priority, reverse=True)
        for rule in sorted_rules:
            for kw in rule.keywords:
                if kw in prompt_lower:
                    return {
                        "target_model": rule.target_model,
                        "routing_strategy": "intent_detected",
                        "matched_domain": rule.target_model,
                        "reason": f"Detected domain keyword '{kw}' in prompt -> routed to '{rule.target_model}'."
                    }

            for regex in rule.regexes:
                if regex.search(prompt_clean):
                    return {
                        "target_model": rule.target_model,
                        "routing_strategy": "intent_detected",
                        "matched_domain": rule.target_model,
                        "reason": f"Matched domain pattern in prompt -> routed to '{rule.target_model}'."
                    }

        if pii_count > 0 and "financial_adapter" in self.rules:
            return {
                "target_model": "financial_adapter",
                "routing_strategy": "intent_detected",
                "matched_domain": "financial_adapter",
                "reason": f"Detected {pii_count} personal data entities under Decree 13/2023/NĐ-CP."
            }

        return {
            "target_model": self.default_base_model,
            "routing_strategy": "fallback_base",
            "matched_domain": "general_conversational",
            "reason": f"No domain-specific intent detected. Falling back to Base Model: '{self.default_base_model}'."
        }

    def get_registered_models(self) -> list[dict[str, Any]]:
        """Returns dynamic model inventory for /v1/models endpoint."""
        models: list[dict[str, Any]] = [
            {
                "id": "auto",
                "type": "smart_intent_router",
                "description": "Dynamic Intent-Based Auto-Router (Dispatches to LoRA or Base dynamically)",
                "strategy": "Dynamic Gateway Routing",
            },
            {
                "id": self.default_base_model,
                "type": "base_foundation_model",
                "description": "Base Foundation Model for General Conversational & Multimodal AI",
                "strategy": "Base Direct Serving",
            },
        ]
        for adapter_name, rule in self.rules.items():
            models.append({
                "id": adapter_name,
                "type": "lora_adapter",
                "description": rule.description,
                "base_model": self.default_base_model,
                "strategy": rule.strategy,
                "keywords_sample": rule.keywords[:5],
            })
        return models


intent_router = IntentRouter()
