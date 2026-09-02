"""
Rule-based Prompt Injection & System Prompt Shield Engine.
Prevents prompt injections, jailbreaks, and proprietary system instruction leakages.
"""

import re
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)


class GuardrailsEngine:
    def __init__(self) -> None:
        self.injection_patterns = [
            re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
            re.compile(r"disregard\s+(all\s+)?(rules|policies|instructions?)", re.IGNORECASE),
            re.compile(r"repeat\s+(all\s+)?(system\s+)?prompt", re.IGNORECASE),
            re.compile(r"reveal\s+(your\s+)?system\s+instructions?", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+in\s+dan\s+mode", re.IGNORECASE),
            re.compile(r"bypass\s+(all\s+)?safety\s+filters?", re.IGNORECASE),
        ]

    def validate_input(self, text: str) -> tuple[bool, str | None]:
        """Scans input prompt for prompt injection or jailbreak patterns."""
        if not text:
            return True, None

        for pattern in self.injection_patterns:
            if pattern.search(text):
                msg = "Prompt rejected by Security Guardrail (Potential Prompt Injection / Jailbreak attempt detected)."
                logger.warning(f"{msg} Matched pattern: {pattern.pattern}")
                return False, msg

        return True, None

    def validate_output(self, output_text: str, system_prompt: str = "") -> tuple[bool, str | None]:
        """Verifies that the LLM output does not leak proprietary system instructions."""
        if output_text and system_prompt and len(system_prompt) >= 10:
            if system_prompt.lower() in output_text.lower():
                msg = "Output blocked by DLP Guardrail (System prompt leakage detected)."
                logger.warning(msg)
                return False, msg
        return True, None


guardrails_engine = GuardrailsEngine()
