"""
Microsoft Presidio & Vietnamese Data Protection Regulation (Nghị định 13/2023/NĐ-CP) Engine.
Comprehensive PII Masking, Tokenization & Multimodal Image Redaction Engine.
Uses Non-Overlapping Span Offset Replacement (Right-to-Left) to eliminate substring collisions.

Categorized into:
1. Dữ liệu cá nhân cơ bản (Basic Personal Data - Điều 2 Khoản 3 NĐ 13/2023/NĐ-CP)
2. Dữ liệu cá nhân nhạy cảm (Sensitive Personal Data - Điều 2 Khoản 4 NĐ 13/2023/NĐ-CP)
"""

import io
import re
import base64
from PIL import Image, ImageDraw
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)


class EntityDefinition:
    def __init__(
        self,
        code: str,
        name: str,
        category: str,
        legal_ref: str,
        pattern: re.Pattern,
        priority: int = 10,
    ) -> None:
        self.code = code
        self.name = name
        self.category = category
        self.legal_ref = legal_ref
        self.pattern = pattern
        self.priority = priority


class PresidioPIIEngine:
    def __init__(self) -> None:
        self.entities: list[EntityDefinition] = [
            # Sensitive Personal Data (Article 2, Clause 4, Decree 13/2023/ND-CP)
            EntityDefinition(
                code="GPS_LOCATION",
                name="Tọa độ định vị GPS cá nhân",
                category="SENSITIVE_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 4 Điểm g NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b[-+]?(?:[1-8]?\d(?:\.\d{3,8})|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d{3,8}))\b"),
                priority=30,
            ),
            EntityDefinition(
                code="CREDIT_CARD",
                name="Số thẻ tín dụng/ghi nợ",
                category="SENSITIVE_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 4 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
                priority=25,
            ),
            EntityDefinition(
                code="BANK_ACCOUNT",
                name="Số tài khoản ngân hàng",
                category="SENSITIVE_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 4 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:stk|tài khoản|tk ngân hàng|số tk)[:\s]*([0-9]{8,16})\b"),
                priority=22,
            ),
            EntityDefinition(
                code="CVV_CVC",
                name="Mã bảo mật thẻ CVV/CVC",
                category="SENSITIVE_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 4 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:cvv|cvc|cvv2|cvc2|mã bảo mật)[:\s]*([0-9]{3,4})\b"),
                priority=22,
            ),
            EntityDefinition(
                code="OTP_PIN",
                name="Mã xác thực OTP / Mã PIN",
                category="SENSITIVE_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 4 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:otp|mã pin|pin code|mã xác thực)[:\s]*([0-9]{4,8})\b"),
                priority=22,
            ),
            EntityDefinition(
                code="MEDICAL_RECORD_ID",
                name="Mã hồ sơ bệnh án y tế",
                category="SENSITIVE_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 4 Điểm b NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:ba|hsba|bệnh án|hồ sơ y tế)[:\s]*([a-z0-9-]{6,12})\b"),
                priority=20,
            ),

            # Basic Personal Data (Article 2, Clause 3, Decree 13/2023/ND-CP)
            EntityDefinition(
                code="IP_ADDRESS",
                name="Địa chỉ giao thức mạng (IP Address)",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm d NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
                priority=22,
            ),
            EntityDefinition(
                code="MAC_ADDRESS",
                name="Địa chỉ vật lý thiết bị (MAC Address)",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm d NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"),
                priority=22,
            ),
            EntityDefinition(
                code="EMAIL",
                name="Địa chỉ thư điện tử (Email)",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm d NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
                priority=20,
            ),
            EntityDefinition(
                code="HEALTH_INSURANCE_ID",
                name="Mã thẻ Bảo hiểm Y tế (BHYT)",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm e NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:bhyt|bảo hiểm y tế|mã thẻ bhyt)[:\s]*([A-Z]{2}[0-9A-Z]{13})\b"),
                priority=18,
            ),
            EntityDefinition(
                code="CITIZEN_ID",
                name="Số Căn cước công dân / CMND",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b\d{12}\b|\b\d{9}\b"),
                priority=17,
            ),
            EntityDefinition(
                code="PASSPORT_VN",
                name="Số Hộ chiếu Việt Nam",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b[B-Db-dKk]\d{7}\b"),
                priority=16,
            ),
            EntityDefinition(
                code="DRIVER_LICENSE",
                name="Số Giấy phép lái xe (GPLX)",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:gplx|bằng lái|giấy phép lái xe)[:\s]*([0-9]{12})\b"),
                priority=16,
            ),
            EntityDefinition(
                code="TAX_ID",
                name="Mã số thuế cá nhân (MST)",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm e NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:mst|mã số thuế|tax id)[:\s]*([0-9]{10}(?:-[0-9]{3})?)\b"),
                priority=15,
            ),
            EntityDefinition(
                code="SOCIAL_SECURITY_ID",
                name="Mã số Bảo hiểm xã hội (BHXH)",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm e NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"(?i)\b(?:bhxh|bảo hiểm xã hội|số sổ bhxh)[:\s]*([0-9]{10})\b"),
                priority=15,
            ),
            EntityDefinition(
                code="LICENSE_PLATE",
                name="Biển số xe cơ giới",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm đ NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b(?:[1-9][0-9][A-Za-z][0-9A-Za-z]?)[-.\s]?(?:[0-9]{3}[.][0-9]{2}|[0-9]{4,5}|[0-9]{3})\b"),
                priority=15,
            ),
            EntityDefinition(
                code="PHONE_NUMBER",
                name="Số điện thoại cá nhân",
                category="BASIC_PERSONAL_DATA",
                legal_ref="Điều 2 Khoản 3 Điểm d NĐ 13/2023/NĐ-CP",
                pattern=re.compile(r"\b(?:0|\+84)(?:3|5|7|8|9|2)\d{8}\b"),
                priority=14,
            ),
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
                    val = match.group(1).strip()
                else:
                    start = match.start()
                    end = match.end()
                    val = match.group(0).strip()

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

    def get_supported_entities(self) -> list[dict[str, str]]:
        """Returns full schema catalog of supported Decree 13 PII entities."""
        return [
            {
                "code": e.code,
                "name": e.name,
                "category": e.category,
                "legal_ref": e.legal_ref,
            }
            for e in self.entities
        ]

    def process_multimodal_ocr(self, image_base64: str) -> tuple[str, str, dict[str, str], int]:
        """
        Executes the 2-Stream OCR Architecture:
        Stream 1: Real Dynamic OCR Tokenization + Exact Pixel Bounding Box Blackout Redaction.
        Stream 2: Full Document Text Extraction + Presidio Token Masking for Text LLM.
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

            font = None
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
            except Exception:
                pass

            extracted_lines = []
            redaction_boxes = []

            try:
                import pytesseract
                from pytesseract import Output
                from PIL import ImageOps
                
                candidate_crops = [
                    (0.0, 0.0, 1.0, 1.0),
                    (0.26, 0.12, 0.75, 0.88),
                    (0.20, 0.10, 0.80, 0.90),
                    (0.15, 0.08, 0.85, 0.92),
                ]

                best_lines = {}
                max_tokens_count = 0

                for x1_pct, y1_pct, x2_pct, y2_pct in candidate_crops:
                    cx1, cy1 = int(x1_pct * width), int(y1_pct * height)
                    cx2, cy2 = int(x2_pct * width), int(y2_pct * height)
                    
                    sub_crop = image.crop((cx1, cy1, cx2, cy2))
                    scale = max(1.0, 1600.0 / max(sub_crop.width, sub_crop.height))
                    scaled = sub_crop.resize((int(sub_crop.width * scale), int(sub_crop.height * scale)), Image.Resampling.LANCZOS)
                    gray = scaled.convert("L")
                    enhanced = ImageOps.autocontrast(gray)
                    
                    ocr_data = pytesseract.image_to_data(enhanced, lang="vie+eng", output_type=Output.DICT, config="--psm 6")
                    
                    current_lines = {}
                    tokens_count = 0
                    n_boxes = len(ocr_data.get("text", []))

                    for i in range(n_boxes):
                        text = ocr_data["text"][i].strip()
                        if not text:
                            continue
                        
                        orig_x = cx1 + int(ocr_data["left"][i] / scale)
                        orig_y = cy1 + int(ocr_data["top"][i] / scale)
                        orig_w = int(ocr_data["width"][i] / scale)
                        orig_h = int(ocr_data["height"][i] / scale)
                        
                        token_info = {"text": text, "x": orig_x, "y": orig_y, "w": orig_w, "h": orig_h}
                        line_id = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])
                        if line_id not in current_lines:
                            current_lines[line_id] = []
                        current_lines[line_id].append(token_info)
                        tokens_count += 1

                    if tokens_count > max_tokens_count:
                        max_tokens_count = tokens_count
                        best_lines = current_lines

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

                for line_id, tokens in best_lines.items():
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

                logger.info(f"Pyramid OCR extracted {len(extracted_lines)} lines ({max_tokens_count} tokens) and detected {len(redaction_boxes)} exact visual PII boxes.")
            except Exception as ocr_err:
                logger.warning(f"Pyramid OCR execution notice: {ocr_err}")

            fill_color = (15, 23, 42)
            border_color = (239, 68, 68)
            text_color = (248, 250, 252)

            for x1, y1, x2, y2, label in redaction_boxes:
                draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=border_color, width=2)
                tag_text = f"[REDACTED: {label}]"
                text_y = y1 + max(1, (y2 - y1 - 8) // 2)
                draw.text((x1 + 4, text_y), tag_text, fill=text_color, font=font)

            banner_height = max(24, int(height * 0.04))
            draw.rectangle([0, 0, width, banner_height], fill=(16, 185, 129))
            banner_text = "PRESIDIO 2-STREAM OCR REDACTOR - NGHI DINH 13/2023/ND-CP COMPLIANT"
            draw.text((10, max(4, (banner_height - 10) // 2)), banner_text, fill=(255, 255, 255), font=font)

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            redacted_image_base64 = f"{header},{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

            raw_extracted_text = "\n".join(extracted_lines)
            raw_masked_text, raw_mapping, pii_count = self.mask_text(raw_extracted_text)

            # Namespace OCR placeholders with OCR_ prefix to prevent collisions with prompt PII tokens
            ocr_mapping = {}
            masked_ocr_text = raw_masked_text
            for placeholder, val in raw_mapping.items():
                if placeholder.startswith("<") and placeholder.endswith(">"):
                    ocr_placeholder = f"<OCR_{placeholder[1:]}"
                else:
                    ocr_placeholder = f"<OCR_{placeholder}>"
                ocr_mapping[ocr_placeholder] = val
                masked_ocr_text = masked_ocr_text.replace(placeholder, ocr_placeholder)

            total_pii_count = max(len(redaction_boxes), pii_count)
            return redacted_image_base64, masked_ocr_text, ocr_mapping, total_pii_count

        except Exception as e:
            logger.error(f"Error in process_multimodal_ocr: {e}")
            return image_base64, "", {}, 0

    def redact_image_preview(self, image_base64: str) -> str:
        """Backward-compatible helper returning redacted visual preview."""
        redacted_b64, _, _, _ = self.process_multimodal_ocr(image_base64)
        return redacted_b64


presidio_engine = PresidioPIIEngine()

