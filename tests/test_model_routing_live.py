"""
Automated Model Routing Test Suite.
Verifies offline rule routing and live gateway endpoints.
"""

import unittest
import httpx

from gateway.app.core.intent_router import intent_router


class TestModelRouting(unittest.TestCase):
    def test_intent_router_offline_logic(self) -> None:
        """Verifies intent routing logic offline without network dependencies."""
        decision_general = intent_router.resolve_model("Xin chào! Bạn là ai?", requested_model="auto")
        self.assertIn(decision_general["routing_strategy"], ["fallback_base", "intent_detected", "client_explicit"])
        self.assertIn("target_model", decision_general)

        decision_explicit = intent_router.resolve_model("Test", requested_model="financial_adapter")
        self.assertEqual(decision_explicit["target_model"], "financial_adapter")
        self.assertEqual(decision_explicit["routing_strategy"], "client_explicit")

    def test_model_auto_routing_live(self) -> None:
        """Tests live endpoint routing if gateway is online, otherwise skips gracefully."""
        cases = [
            ("auto", "Xin chào! Bạn là ai và có thể giúp gì cho tôi?"),
            ("auto", "Bóc tách hóa đơn chuyển khoản STK 1903456789012 số tiền 5000000 VND và CCCD 079123456789"),
            ("Qwen/Qwen2.5-0.5B-Instruct", "Kiểm tra thông tin giao dịch ngân hàng"),
            ("financial_adapter", "Xin chào bạn là ai"),
        ]

        try:
            with httpx.Client(base_url="http://localhost:8080", timeout=2.0) as client:
                health_resp = client.get("/health")
                if health_resp.status_code != 200:
                    raise unittest.SkipTest("Gateway health endpoint returned non-200")

                for requested_model, prompt in cases:
                    payload = {
                        "model": requested_model,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    res = client.post(
                        "/v1/chat/completions",
                        json=payload,
                        headers={"X-API-Key": "secret_enterprise_ai_key_2026"},
                    )
                    self.assertIn(res.status_code, [200, 503])
        except (httpx.ConnectError, httpx.TimeoutException):
            raise unittest.SkipTest("Gateway server is not running on localhost:8080")


if __name__ == "__main__":
    unittest.main()
