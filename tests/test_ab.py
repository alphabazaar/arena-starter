"""The Alpha Bazaar client against a local fake: auth, retries, rate limits, error messages and key redaction."""
import unittest
from unittest import mock

from agent.ab import AB, ABError, redact
from tests.fakes import TEST_KEY, FakeServer


class Client(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("agent.ab.time.sleep")  # no real waiting in tests
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, routes):
        server = FakeServer(routes)
        self.addCleanup(server.close)
        return server, AB(server.url, TEST_KEY)

    def test_public_reads_send_no_key(self):
        server, ab = self.serve({("GET", "/api/market/pulse"): (200, {"market_status": "open"})})
        self.assertEqual(ab.get("/api/market/pulse")["market_status"], "open")
        self.assertNotIn("authorization", server.requests[0]["headers"])

    def test_auth_sends_bearer_key(self):
        server, ab = self.serve({("GET", "/api/agents/me"): (200, {"agent": {"username": "x"}})})
        ab.get("/api/agents/me", auth=True)
        self.assertEqual(server.requests[0]["headers"]["authorization"], f"Bearer {TEST_KEY}")

    def test_auth_without_key_fails_before_calling(self):
        server, _ = self.serve({})
        with self.assertRaisesRegex(ABError, "AB_API_KEY"):
            AB(server.url).get("/api/agents/me", auth=True)
        self.assertEqual(server.requests, [])

    def test_waits_out_a_rate_limit_then_retries(self):
        server, ab = self.serve({("GET", "/api/arena/me"): [
            (429, {"detail": "Rate limit exceeded (60 per 1 minute)."}, {"Retry-After": "2"}),
            (200, {"ok": True})]})
        self.assertEqual(ab.get("/api/arena/me", auth=True), {"ok": True})
        self.assertEqual(len(server.requests), 2)
        self.sleep.assert_any_call(3.0)  # Retry-After plus a second

    def test_does_not_wait_an_hour(self):
        server, ab = self.serve({("POST", "/api/agents/register"): (
            429, {"detail": "Registration rate limit reached (5 per 1 hour)."}, {"Retry-After": "3600"})})
        with self.assertRaises(ABError) as caught:
            ab.post("/api/agents/register", {"name": "x"})
        self.assertEqual(caught.exception.status, 429)
        self.assertIn("Try again in 3600 seconds.", str(caught.exception))
        self.assertEqual(len(server.requests), 1)

    def test_a_429_without_retry_after_is_not_retried(self):
        server, ab = self.serve({("POST", "/api/publisher/submit-pick"): (
            429, {"detail": "Weekly signal limit reached (50 signals/week)"})})
        with self.assertRaisesRegex(ABError, "Weekly signal limit"):
            ab.post("/api/publisher/submit-pick", {"ticker": "NVDA"})
        self.assertEqual(len(server.requests), 1)

    def test_reads_retry_server_errors(self):
        server, ab = self.serve({("GET", "/api/market/pulse"): [(503, {"detail": "busy"}), (200, {"ok": 1})]})
        self.assertEqual(ab.get("/api/market/pulse"), {"ok": 1})
        self.assertEqual(len(server.requests), 2)

    def test_writes_never_retry_server_errors(self):
        server, ab = self.serve({("POST", "/api/publisher/submit-pick"): (500, {"detail": "oops"})})
        with self.assertRaises(ABError):
            ab.post("/api/publisher/submit-pick", {"ticker": "NVDA"})
        self.assertEqual(len(server.requests), 1)  # it may have gone through: never publish twice

    def test_field_errors_are_one_readable_line(self):
        server, ab = self.serve({("POST", "/api/agents/me/log"): (422, {"detail": [
            {"loc": ["body", "content"], "msg": "String should have at most 4000 characters"},
            {"loc": ["body", "ticker"], "msg": "Invalid ticker symbol"}]})})
        with self.assertRaises(ABError) as caught:
            ab.post("/api/agents/me/log", {"content": "x"})
        self.assertIn("content: String should have at most 4000 characters; ticker: Invalid ticker symbol",
                      str(caught.exception))

    def test_401_explains_the_key(self):
        server, ab = self.serve({("GET", "/api/agents/me"): (401, {"detail": "Invalid API key"})})
        with self.assertRaisesRegex(ABError, "Check AB_API_KEY"):
            ab.get("/api/agents/me", auth=True)

    def test_keys_in_error_text_are_masked(self):
        server, ab = self.serve({("GET", "/api/agents/me"): (400, {"detail": f"bad key {TEST_KEY}"})})
        with self.assertRaises(ABError) as caught:
            ab.get("/api/agents/me", auth=True)
        self.assertNotIn(TEST_KEY, str(caught.exception))
        self.assertIn("ab_live_***", str(caught.exception))

    def test_a_read_that_times_out_is_not_retried(self):
        import requests
        ab = AB("http://127.0.0.1:9", TEST_KEY)
        with mock.patch.object(ab.session, "request", side_effect=requests.ReadTimeout("slow")) as request:
            with self.assertRaisesRegex(ABError, "no answer from .* within 30 seconds"):
                ab.get("/api/market/pulse")
        self.assertEqual(request.call_count, 1)

    def test_unreachable_api(self):
        ab = AB("http://127.0.0.1:9", TEST_KEY, timeout=2)
        with self.assertRaisesRegex(ABError, "couldn't reach"):
            ab.get("/api/market/pulse")


class Redact(unittest.TestCase):
    def test_masks_every_kind_of_key(self):
        text = redact(f"{TEST_KEY} sk-ant-api03-abcdefghijk AIzaSyA1234567890abcdefghijk Bearer eyJhbGciOi.abc.def")
        for secret in (TEST_KEY, "sk-ant-api03-abcdefghijk", "AIzaSyA1234567890abcdefghijk", "eyJhbGciOi"):
            self.assertNotIn(secret, text)

    def test_leaves_normal_text_alone(self):
        text = "Bought NVDA: sk-ip the noise, up 3% (risk-on)."
        self.assertEqual(redact(text), text)


if __name__ == "__main__":
    unittest.main()
