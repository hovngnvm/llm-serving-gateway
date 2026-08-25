"""
Structured Output Parser & Schema Validator.
Parses raw LLM markdown text into clean JSON and validates data integrity using Pydantic schemas.
Replaces arbitrary heuristic if/else checks with standard declarative data contracts.
"""

import json
import re
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator, ValidationError
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False


class FinancialTransactionSchema(BaseModel):
    """Declarative schema contract for extracted financial transactions."""

    transaction_type: str | None = Field(default="TRANSACTION")
    amount: float | None = Field(default=None, ge=0.0)
    currency: Literal["VND", "USD", "EUR", "JPY", ""] | None = "VND"
    sender_name: str | None = None
    sender_account: str | None = None
    receiver_name: str | None = None
    receiver_account: str | None = None
    subtotal: float | None = Field(default=None, ge=0.0)
    tax: float | None = Field(default=None, ge=0.0)
    total: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def verify_math_balance(self) -> "FinancialTransactionSchema":
        """Declaratively verifies arithmetic consistency when line items are present."""
        if self.subtotal is not None and self.tax is not None and self.total is not None:
            expected = round(self.subtotal + self.tax, 2)
            if abs(expected - round(self.total, 2)) > 0.01:
                raise ValueError(
                    f"Arithmetic imbalance: subtotal ({self.subtotal}) + tax ({self.tax}) = {expected}, but total is {self.total}"
                )
        return self


class OutputValidator:
    """Enterprise Output Parser & Validator."""

    def __init__(self) -> None:
        self.validation_count = 0
        self.auto_repair_count = 0

    def extract_json_string(self, text: str) -> str | None:
        """Extracts JSON substring from markdown code blocks or raw strings."""
        if not text or not text.strip():
            return None

        cleaned = text.strip()
        if "```json" in cleaned:
            parts = cleaned.split("```json")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()

        if cleaned.startswith("{") or cleaned.startswith("["):
            return cleaned

        return None

    def _fallback_repair(self, text: str) -> dict[str, Any] | None:
        """Heuristic syntax repair for missing closing braces and trailing commas."""
        cleaned = text.strip()
        cleaned = re.sub(r",\s*$", "", cleaned)
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        open_braces = cleaned.count("{")
        close_braces = cleaned.count("}")
        if open_braces > close_braces:
            cleaned += "}" * (open_braces - close_braces)

        open_brackets = cleaned.count("[")
        close_brackets = cleaned.count("]")
        if open_brackets > close_brackets:
            cleaned += "]" * (open_brackets - close_brackets)

        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        try:
            return json.loads(cleaned)
        except Exception:
            return None

    def parse_and_validate(
        self,
        raw_text: str,
        schema_class: type[BaseModel] | None = None,
    ) -> tuple[dict[str, Any] | None, bool, list[str], bool]:
        """
        Extracts, parses, repairs, and optionally validates LLM output against a Pydantic schema.
        When schema_class is None, acts as a generic domain-agnostic JSON parser and syntax repairer.
        Returns: (structured_dict, is_schema_valid, list_of_error_messages, was_syntax_repaired)
        """
        json_str = self.extract_json_string(raw_text)
        if not json_str:
            return None, False, ["No JSON structure detected in output."], False

        raw_dict = None
        was_repaired = False

        # 1. Standard JSON Parse
        try:
            raw_dict = json.loads(json_str)
        except json.JSONDecodeError:
            # 2. Syntax Auto-Repair if malformed
            if HAS_JSON_REPAIR:
                try:
                    repaired = json_repair.repair_json(json_str, return_objects=True)
                    if isinstance(repaired, (dict, list)):
                        raw_dict = repaired
                        was_repaired = True
                        self.auto_repair_count += 1
                except Exception:
                    pass

            if raw_dict is None:
                repaired = self._fallback_repair(json_str)
                if isinstance(repaired, (dict, list)):
                    raw_dict = repaired
                    was_repaired = True
                    self.auto_repair_count += 1

        if not isinstance(raw_dict, dict):
            if isinstance(raw_dict, list):
                raw_dict = {"items": raw_dict}
            else:
                return None, False, ["Failed to decode valid JSON object."], was_repaired

        # 3. Optional Declarative Pydantic Schema Validation
        if schema_class is not None:
            try:
                validated_model = schema_class.model_validate(raw_dict)
                self.validation_count += 1
                return validated_model.model_dump(exclude_none=True), True, [], was_repaired
            except ValidationError as err:
                error_messages: list[str] = []
                for e in err.errors():
                    loc_str = ".".join(str(x) for x in e.get("loc", ())) if e.get("loc") else "schema"
                    error_messages.append(f"{loc_str}: {e['msg']}")
                logger.warning(f"Schema validation warning: {'; '.join(error_messages)}")
                return raw_dict, False, error_messages, was_repaired

        # Generic valid dictionary
        self.validation_count += 1
        return raw_dict, True, [], was_repaired


output_validator = OutputValidator()
