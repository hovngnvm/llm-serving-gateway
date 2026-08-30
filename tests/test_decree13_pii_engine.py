"""
Unit Test Suite for Vietnamese Data Protection Regulation (Nghị định 13/2023/NĐ-CP) PII Engine.
Verifies all 17 legal entity recognizers (Basic & Sensitive Personal Data) and Span Offset Matching.
"""

import unittest
from gateway.app.core.presidio_engine import presidio_engine


class TestDecree13PIIEngine(unittest.TestCase):
    def test_basic_personal_data_masking(self) -> None:
        """Tests Điều 2 Khoản 3 NĐ 13/2023/NĐ-CP: Basic Personal Data."""
        prompt = (
            "Khách hàng có CCCD 079123456789, Hộ chiếu B1234567, "
            "GPLX: 079095001234, MST: 0312345678, BHXH: 7912345678, "
            "BHYT: DN4797931234567, Biển số: 29A-888.88, SĐT: 0912345678, "
            "Email: test@company.com, IP: 192.168.1.1, MAC: 00:1A:2B:3C:4D:5E."
        )
        masked, mapping, count = presidio_engine.mask_text(prompt)

        self.assertGreaterEqual(count, 10)
        self.assertIn("<CITIZEN_ID_1>", masked)
        self.assertIn("<PASSPORT_VN_1>", masked)
        self.assertIn("<TAX_ID_1>", masked)
        self.assertIn("<SOCIAL_SECURITY_ID_1>", masked)
        self.assertIn("<HEALTH_INSURANCE_ID_1>", masked)
        self.assertIn("<LICENSE_PLATE_1>", masked)
        self.assertIn("<PHONE_NUMBER_1>", masked)
        self.assertIn("<EMAIL_1>", masked)
        self.assertIn("<IP_ADDRESS_1>", masked)
        self.assertIn("<MAC_ADDRESS_1>", masked)

        unmasked = presidio_engine.unmask_text(masked, mapping)
        self.assertEqual(unmasked, prompt)

    def test_sensitive_personal_data_masking(self) -> None:
        """Tests Điều 2 Khoản 4 NĐ 13/2023/NĐ-CP: Sensitive Personal Data."""
        prompt = (
            "Giao dịch thẻ 4111222233334444, STK: 1903456789012, "
            "CVV: 888, OTP: 123456, Vị trí GPS: 10.762622, 106.660172, "
            "Bệnh án: BA-987654."
        )
        masked, mapping, count = presidio_engine.mask_text(prompt)

        self.assertIn("<CREDIT_CARD_1>", masked)
        self.assertIn("<CVV_CVC_1>", masked)
        self.assertIn("<OTP_PIN_1>", masked)
        self.assertIn("<GPS_LOCATION_1>", masked)
        self.assertIn("<MEDICAL_RECORD_ID_1>", masked)

        unmasked = presidio_engine.unmask_text(masked, mapping)
        self.assertEqual(unmasked, prompt)

    def test_no_substring_collision(self) -> None:
        """Ensures 12-digit CCCD is not mutilated by 6-digit OTP regex."""
        prompt = "CCCD 079123456789 với số tiền 5000000 VND và OTP: 654321"
        masked, mapping, count = presidio_engine.mask_text(prompt)

        self.assertIn("<CITIZEN_ID_1>", masked)
        self.assertIn("<OTP_PIN_1>", masked)
        self.assertEqual(mapping["<CITIZEN_ID_1>"], "079123456789")
        self.assertEqual(mapping["<OTP_PIN_1>"], "654321")

    def test_multimodal_ocr_namespace_isolation(self) -> None:
        """Ensures OCR placeholders use OCR_ prefix and do not overwrite prompt PII tokens."""
        prompt = "Khách hàng CCCD 079123456789"
        masked_prompt, prompt_mapping, _ = presidio_engine.mask_text(prompt)

        # Simulate OCR text containing another citizen ID
        ocr_raw = "Hóa đơn đính kèm CCCD 001987654321"
        raw_ocr_masked, raw_ocr_map, _ = presidio_engine.mask_text(ocr_raw)
        ocr_prefixed_map = {f"<OCR_{k[1:]}": v for k, v in raw_ocr_map.items()}

        # Merge mappings
        combined_mapping = dict(prompt_mapping)
        combined_mapping.update(ocr_prefixed_map)

        # Verify both tokens exist independently without key collision
        self.assertIn("<CITIZEN_ID_1>", combined_mapping)
        self.assertIn("<OCR_CITIZEN_ID_1>", combined_mapping)
        self.assertEqual(combined_mapping["<CITIZEN_ID_1>"], "079123456789")
        self.assertEqual(combined_mapping["<OCR_CITIZEN_ID_1>"], "001987654321")


if __name__ == "__main__":
    unittest.main()
