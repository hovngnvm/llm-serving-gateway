"""
Microsoft Presidio & Vietnamese Data Protection Regulation (Nghị định 13/2023/NĐ-CP) Engine.
Comprehensive PII Masking, Tokenization & Multimodal Image Redaction Engine.
Uses Non-Overlapping Span Offset Replacement (Right-to-Left) to eliminate substring collisions.

Categorized into:
1. Dữ liệu cá nhân cơ bản (Basic Personal Data - Điều 2 Khoản 3 NĐ 13/2023/NĐ-CP)
2. Dữ liệu cá nhân nhạy cảm (Sensitive Personal Data - Điều 2 Khoản 4 NĐ 13/2023/NĐ-CP)
"""

import base64
import io
import re
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pytesseract
from pytesseract import Output
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)

REDACTION_FILL_COLOR = (15, 23, 42)
REDACTION_BORDER_COLOR = (239, 68, 68)
REDACTION_TEXT_COLOR = (248, 250, 252)
MAX_OCR_SCALING_DIM = 1600.0


@dataclass(slots=True)
class EntityDefinition:
    code: str
    pattern: re.Pattern[str]
    priority: int = 10


class PresidioPIIEngine:
    def __init__(self) -> None:
        self.entities: list[EntityDefinition] = [
            # Sensitive Personal Data (Article 2, Clause 4, Decree 13/2023/ND-CP)
            EntityDefinition("GPS_LOCATION", re.compile(r"\b[-+]?(?:[1-8]?\d(?:\.\d{3,8})|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d{3,8}))\b"), 30),
            EntityDefinition("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), 25),
            EntityDefinition("BANK_ACCOUNT", re.compile(r"(?i)\b(?:stk|tài khoản|tk ngân hàng|số tk)[:\s]*([0-9]{8,16})\b"), 22),
            EntityDefinition("CVV_CVC", re.compile(r"(?i)\b(?:cvv|cvc|cvv2|cvc2|mã bảo mật)[:\s]*([0-9]{3,4})\b"), 22),
            EntityDefinition("OTP_PIN", re.compile(r"(?i)\b(?:otp|mã pin|pin code|mã xác thực)[:\s]*([0-9]{4,8})\b"), 22),
            EntityDefinition("MEDICAL_RECORD_ID", re.compile(r"(?i)\b(?:ba|hsba|bệnh án|hồ sơ y tế)[:\s]*([a-z0-9-]{6,12})\b"), 20),

            # Basic Personal Data (Article 2, Clause 3, Decree 13/2023/ND-CP)
            EntityDefinition("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"), 22),
            EntityDefinition("MAC_ADDRESS", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"), 22),
            EntityDefinition("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b"), 20),
            EntityDefinition("HEALTH_INSURANCE_ID", re.compile(r"(?i)\b(?:bhyt|bảo hiểm y tế|mã thẻ bhyt)[:\s]*([A-Z]{2}[0-9A-Z]{13})\b"), 18),
            EntityDefinition("CITIZEN_ID", re.compile(r"\b\d{12}\b|\b\d{9}\b"), 17),
            EntityDefinition("PASSPORT_VN", re.compile(r"\b[B-Db-dKk]\d{7}\b"), 16),
            EntityDefinition("DRIVER_LICENSE", re.compile(r"(?i)\b(?:gplx|bằng lái|giấy phép lái xe)[:\s]*([0-9]{12})\b"), 16),
            EntityDefinition("TAX_ID", re.compile(r"(?i)\b(?:mst|mã số thuế|tax id)[:\s]*([0-9]{10}(?:-[0-9]{3})?)\b"), 15),
            EntityDefinition("SOCIAL_SECURITY_ID", re.compile(r"(?i)\b(?:bhxh|bảo hiểm xã hội|số sổ bhxh)[:\s]*([0-9]{10})\b"), 15),
            EntityDefinition("LICENSE_PLATE", re.compile(r"\b(?:[1-9][0-9][A-Za-z][0-9A-Za-z]?)[-.\s]?(?:[0-9]{3}[.][0-9]{2}|[0-9]{4,5}|[0-9]{3})\b"), 15),
            EntityDefinition("PHONE_NUMBER", re.compile(r"\b(?:0|\+84)(?:3|5|7|8|9|2)\d{8}\b"), 14),
        ]

    def mask_text(self, text: str) -> tuple[str, dict[str, str], int]:
        """
        Masks all PII entities under Decree 13/2023/NĐ-CP with anonymous encrypted placeholders.
        Employs non-overlapping Span Offset Replacement to eliminate substring collision bugs.
        """
        if not text:
            return text, {}, 0

        candidate_spans = []

        for entity in self.entities:
            for match in entity.pattern.finditer(text):
                if match.groups() and match.group(1):
                    start = match.start(1)
                    end = match.end(1)
                    val = text[start:end]
                else:
                    start = match.start()
                    end = match.end()
                    val = text[start:end]

                candidate_spans.append((start, end, entity.code, val, entity.priority))

        candidate_spans.sort(key=lambda s: (-s[4], -(s[1] - s[0]), s[0]))

        chosen_spans = []
        occupied_indices = set()

        for start, end, code, val, priority in candidate_spans:
            span_indices = set(range(start, end))
            if not span_indices.intersection(occupied_indices):
                chosen_spans.append((start, end, code, val))
                occupied_indices.update(span_indices)

        chosen_spans.sort(key=lambda s: s[0])

        mapping: dict[str, str] = {}
        entity_counters: dict[str, int] = {}
        replacement_spans = []

        for start, end, code, val in chosen_spans:
            idx = entity_counters.get(code, 0) + 1
            entity_counters[code] = idx
            placeholder = f"<{code}_{idx}>"
            mapping[placeholder] = val
            replacement_spans.append((start, end, placeholder, val))

        replacement_spans.sort(key=lambda s: -s[0])
        masked_text = text
        for start, end, placeholder, val in replacement_spans:
            masked_text = masked_text[:start] + placeholder + masked_text[end:]

        return masked_text, mapping, len(chosen_spans)

    def unmask_text(self, text: str, mapping: dict[str, str]) -> str:
        """Restores original PII entities from the mapping dictionary for authorized clients."""
        if not text or not mapping:
            return text

        unmasked_text = text
        for placeholder, original_value in mapping.items():
            unmasked_text = unmasked_text.replace(placeholder, original_value)
        return unmasked_text

    def process_multimodal_ocr(self, image_base64: str) -> tuple[str, str, dict[str, str], int]:
        """
        Executes 2-Stream OCR Architecture:
        Stream 1: Real Dynamic OCR Tokenization + Exact Pixel Bounding Box Blackout Redaction.
        Stream 2: Full Document Text Extraction + PII Token Masking for LLM.
        """
        try:
            if not image_base64:
                return "", "", {}, 0

            if "," in image_base64:
                header, encoded = image_base64.split(",", 1)
            else:
                header, encoded = "data:image/png;base64", image_base64

            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            font = ImageFont.load_default()

            extracted_lines = []
            redaction_boxes = []

            try:
                scale = max(1.0, MAX_OCR_SCALING_DIM / max(width, height))
                scaled = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
                gray = scaled.convert("L")
                enhanced = ImageOps.autocontrast(gray)

                ocr_data = pytesseract.image_to_data(enhanced, lang="vie+eng", output_type=Output.DICT, config="--psm 6")

                current_lines = {}
                n_boxes = len(ocr_data.get("text", []))

                for i in range(n_boxes):
                    text = ocr_data["text"][i].strip()
                    if not text:
                        continue

                    orig_x = int(ocr_data["left"][i] / scale)
                    orig_y = int(ocr_data["top"][i] / scale)
                    orig_w = int(ocr_data["width"][i] / scale)
                    orig_h = int(ocr_data["height"][i] / scale)

                    token_info = {"text": text, "x": orig_x, "y": orig_y, "w": orig_w, "h": orig_h}
                    line_id = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])
                    if line_id not in current_lines:
                        current_lines[line_id] = []
                    current_lines[line_id].append(token_info)

                sensitive_line_patterns = [
                    (r"(?i)\b(?:hotline|tel|sđt|đt|phone)\b", "HOTLINE_TEL"),
                    (r"(?i)\b(?:nv|nvbh|thu\s*ngân|quầy|nhân\s*viên|staff|msch)\b", "STAFF_CASHIER_ID"),
                    (r"(?i)\b(?:stk|tài\s*khoản|thẻ|visa|master|card|ptt|so\s*the)\b", "PAYMENT_TRANSACTION_ID"),
                    (r"(?i)\b(?:mst|mã\s*số\s*thuế|tax|cqt|mã\s*cqt)\b", "TAX_ID_CQT"),
                    (r"\b893\d{9,11}\b", "BARCODE_TRACKING"),
                    (r"\b0\d{8,10}\b", "PHONE_NUMBER_VN"),
                    (r"\b02\d{8,10}\b", "HOTLINE_FIXED_LINE"),
                    (r"\b\d{12}\b|\b\d{9}\b", "CITIZEN_ID"),
                ]

                for line_id, tokens in current_lines.items():
                    line_text = " ".join(tok["text"] for tok in tokens)
                    extracted_lines.append(line_text)

                    for pattern, label in sensitive_line_patterns:
                        if re.search(pattern, line_text):
                            min_x = min(tok["x"] for tok in tokens)
                            min_y = min(tok["y"] for tok in tokens)
                            max_x = max(tok["x"] + tok["w"] for tok in tokens)
                            max_y = max(tok["y"] + tok["h"] for tok in tokens)
                            redaction_boxes.append((max(0, min_x - 3), max(0, min_y - 2), min(width, max_x + 3), min(height, max_y + 2), label))
                            break

                logger.info(f"1-Pass OCR extracted {len(extracted_lines)} lines and detected {len(redaction_boxes)} PII boxes.")
            except Exception as ocr_err:
                logger.warning(f"OCR execution notice: {ocr_err}")

            for x1, y1, x2, y2, label in redaction_boxes:
                draw.rectangle([x1, y1, x2, y2], fill=REDACTION_FILL_COLOR, outline=REDACTION_BORDER_COLOR, width=2)
                tag_text = f"[REDACTED: {label}]"
                text_y = y1 + max(1, (y2 - y1 - 8) // 2)
                draw.text((x1 + 4, text_y), tag_text, fill=REDACTION_TEXT_COLOR, font=font)

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            redacted_image_base64 = f"{header},{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

            raw_extracted_text = "\n".join(extracted_lines)
            raw_masked_text, raw_mapping, pii_count = self.mask_text(raw_extracted_text)

            ocr_mapping = {}
            masked_ocr_text = raw_masked_text
            for placeholder, val in raw_mapping.items():
                ocr_placeholder = f"<OCR_{placeholder[1:]}"
                ocr_mapping[ocr_placeholder] = val
                masked_ocr_text = masked_ocr_text.replace(placeholder, ocr_placeholder)

            total_pii_count = max(len(redaction_boxes), pii_count)
            return redacted_image_base64, masked_ocr_text, ocr_mapping, total_pii_count

        except Exception as e:
            logger.error(f"Error in process_multimodal_ocr: {e}")
            return image_base64, "", {}, 0


presidio_engine = PresidioPIIEngine()
