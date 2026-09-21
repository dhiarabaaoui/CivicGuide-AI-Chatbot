from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from apps.rag_api import app


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_frontend_is_served_from_root(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("CivicGuide", response.text)
        self.assertIn('id="composer"', response.text)
        self.assertIn('id="evidencePanel"', response.text)
        self.assertIn('id="historyList"', response.text)
        self.assertNotIn("rag_candidate_structural", response.text)
        self.assertNotIn("Système prêt", response.text)
        self.assertIn("donnée personnelle confidentielle", response.text)
        self.assertTrue(response.headers.get("x-request-id"))
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")

    def test_static_assets_are_available(self) -> None:
        css = self.client.get("/static/styles.css")
        javascript = self.client.get("/static/app.js")
        self.assertEqual(css.status_code, 200)
        self.assertEqual(javascript.status_code, 200)
        self.assertIn("@media (max-width: 820px)", css.text)
        self.assertIn("--page: #f7f8fb", css.text)
        self.assertIn(".evidence-drawer", css.text)
        self.assertIn("async function submitQuestion", javascript.text)
        self.assertIn("async function runDemo", javascript.text)
        self.assertIn("function persistCurrentConversation", javascript.text)
        self.assertIn("async function sendFeedback", javascript.text)
        self.assertIn('fetch("/v1/feedback"', javascript.text)
        self.assertIn("localStorage.setItem", javascript.text)
        self.assertIn("Démo en cours", javascript.text)

    def test_operational_health_and_feedback_endpoints(self) -> None:
        live = self.client.get("/health/live")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json(), {"status": "alive"})

        feedback = self.client.post(
            "/v1/feedback",
            json={"trace_id": "trace_12345678", "rating": "helpful"},
        )
        self.assertEqual(feedback.status_code, 202)
        self.assertEqual(feedback.json()["status"], "accepted")
        self.assertEqual(feedback.headers["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()
