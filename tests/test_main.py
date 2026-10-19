"""Whole runs against a fake Alpha Bazaar API: what gets published, what gets logged, and what never leaks."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import main as agent_main
from agent.providers import ProviderError
from tests.fakes import TEST_KEY, fake_ab

SUBMIT = "/api/publisher/submit-pick"
LOG = "/api/agents/me/log"


class Run(unittest.TestCase):
    def setUp(self):
        for target in ("agent.ab.time.sleep", "agent.config.load_dotenv"):  # no waiting, and never a real .env
            patcher = mock.patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.logs = Path(tempfile.mkdtemp())

    def run_agent(self, server, *args, key=TEST_KEY, **environment):
        env = {"AB_API_BASE": server.url, "LLM_PROVIDER": "rules", **environment}
        if key:
            env["AB_API_KEY"] = key
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(out):
            code = agent_main.main(["--log-dir", str(self.logs), *args])
        return code, out.getvalue()

    def serve(self, **overrides):
        server = fake_ab(**overrides)
        self.addCleanup(server.close)
        return server

    def assert_no_key_anywhere(self, output):
        self.assertNotIn(TEST_KEY, output)
        for file in self.logs.iterdir():
            self.assertNotIn(TEST_KEY, file.read_text(encoding="utf-8"), file.name)

    def test_live_run_publishes_one_signal_and_logs_why(self):
        server = self.serve()
        code, output = self.run_agent(server)
        self.assertEqual(code, 0, output)

        [submit] = server.calls("POST", SUBMIT)
        body = submit["body"]
        # KGC is already held, so the momentum rules pick NIC (actionable, +9% since detection)
        self.assertEqual({k: body[k] for k in ("ticker", "action", "time_horizon", "position_size",
                                               "auto_close_days", "confidence")},
                         {"ticker": "NIC", "action": "buy", "time_horizon": "1W", "position_size": 5.0,
                          "auto_close_days": 7, "confidence": 6})
        self.assertEqual((body["stop_loss"], body["price_target"]), (149.59, 186.99))  # 8% / 15% from $162.60
        self.assertTrue(body["reasoning"].startswith("Momentum rule: NIC is up 9.0%"))
        self.assertEqual(submit["headers"]["authorization"], f"Bearer {TEST_KEY}")

        [entry] = server.calls("POST", LOG)
        log = entry["body"]
        self.assertEqual((log["kind"], log["ticker"]), ("decision", "NIC"))
        self.assertTrue(log["content"].startswith(
            "Bought NIC: 5% of the paper portfolio for up to 7 days (1W), confidence 6/10, stop $149.59, target "
            "$186.99. Context: market open · SPY +0.42% · QQQ +1.21% · DIA -0.03%;"), log["content"])
        self.assertIn("Reasoning: Momentum rule: NIC is up 9.0%", log["content"])
        self.assertEqual(log["meta"]["signal_id"], "11111111-2222-4333-8444-555555555555")
        self.assertEqual((log["meta"]["provider"], log["meta"]["model"]), ("rules", "rules-based (no LLM)"))
        self.assertLessEqual(len(json.dumps(log["meta"]).encode()), 2048)

        # The profile's model label follows what's running, and the agent joins the open season it wasn't in
        self.assertEqual(server.calls("PATCH", "/api/agents/me")[0]["body"], {"model": "rules-based (no LLM)"})
        self.assertEqual(len(server.calls("POST", "/api/arena/join")), 1)
        self.assertIn("Attestation: AB-new123", output)
        self.assert_no_key_anywhere(output)
        record = json.loads(next(self.logs.glob("run-*.json")).read_text())
        self.assertEqual(record["signal"]["id"], "11111111-2222-4333-8444-555555555555")
        self.assertEqual(record["run_log"]["seq"], 7)

    def test_dry_run_writes_nothing(self):
        server = self.serve()
        code, output = self.run_agent(server, "--dry-run")
        self.assertEqual(code, 0, output)
        self.assertEqual([r for r in server.requests if r["method"] != "GET"], [])
        self.assertIn("Would publish: buy NIC near $162.60 · stop $149.59 · target $186.99", output)
        self.assert_no_key_anywhere(output)

    def test_dry_run_works_without_a_key(self):
        server = self.serve()
        code, output = self.run_agent(server, "--dry-run", key=None)
        self.assertEqual(code, 0, output)
        self.assertEqual({r["path"] for r in server.requests}, {
            "/api/market/pulse", "/api/feed/daily-digest", "/api/confluence/smart-feed",
            "/api/sectors/intelligence", "/api/arena/seasons"})
        self.assertTrue(all("authorization" not in r["headers"] for r in server.requests))
        self.assertIn("Decision: BUY NIC", output)  # nothing is held without a key

    def test_live_run_needs_a_key(self):
        server = self.serve()
        code, output = self.run_agent(server, key=None)
        self.assertEqual(code, 2)
        self.assertIn("register.py", output)
        self.assertEqual(server.requests, [])

    def test_bad_key_stops_before_anything_else(self):
        server = self.serve(me=(401, {"detail": "Invalid or revoked API key"}))
        code, output = self.run_agent(server)
        self.assertEqual(code, 1)
        self.assertIn("Check AB_API_KEY", output)
        self.assertEqual(len(server.requests), 1)

    def test_stale_levels_are_redone_from_the_live_price(self):
        rejected = (400, {"detail": "stop_loss must be below current price ($148.00) for buy signals. Got $149.59."})
        server = self.serve()
        accept = server.routes[("POST", SUBMIT)]
        calls = []

        def submit(request):  # AB's live price ($148) moved under our stop: rejected once, then accepted
            calls.append(request["body"])
            return rejected if len(calls) == 1 else accept(request)

        server.routes[("POST", SUBMIT)] = submit
        code, output = self.run_agent(server)
        self.assertEqual(code, 0, output)
        self.assertEqual(len(calls), 2)
        self.assertEqual((calls[1]["stop_loss"], calls[1]["price_target"]), (136.16, 170.2))  # from $148.00

    def test_rejected_signal_is_logged_and_flagged(self):
        server = self.serve(submit=(429, {"detail": "Weekly signal limit reached (50 signals/week)"}))
        code, output = self.run_agent(server)
        self.assertEqual(code, 1)
        [entry] = server.calls("POST", LOG)
        self.assertIn("Tried to buy NIC, but Alpha Bazaar rejected the signal: Weekly signal limit",
                      entry["body"]["content"])

    def test_full_portfolio_means_no_trade(self):
        server = self.serve(picks=(200, {"picks": [{"id": f"p{i}", "ticker": f"T{i}", "status": "open",
                                                    "position_size": 10.0} for i in range(10)], "has_more": False}))
        code, output = self.run_agent(server)
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        [entry] = server.calls("POST", LOG)
        self.assertTrue(entry["body"]["content"].startswith("No trade today: wanted to buy NIC, but the paper "
                                                            "portfolio is fully allocated"))
        self.assertEqual(entry["body"]["meta"]["reason"], "no_room")

    def test_invalid_model_answer_means_no_trade(self):
        server = self.serve()
        with mock.patch("agent.providers.Rules.decide", return_value='{"action": "buy", "ticker": "NIC", '
                                                                       '"confidence": 0.9, "reasoning": "sure"}'):
            code, output = self.run_agent(server)
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        [entry] = server.calls("POST", LOG)
        self.assertIn("didn't pass validation", entry["body"]["content"])
        self.assertIn('"confidence" must be a whole number', entry["body"]["content"])

    def test_model_outage_is_logged_but_not_a_failure(self):
        server = self.serve()
        with mock.patch("agent.providers.Rules.decide", side_effect=ProviderError("Gemini API error 503")):
            code, output = self.run_agent(server)
        self.assertEqual(code, 0, output)
        [entry] = server.calls("POST", LOG)
        self.assertEqual(entry["body"]["meta"]["reason"], "model_error")

    def test_bad_model_key_fails_the_run(self):
        server = self.serve()
        with mock.patch("agent.providers.Rules.decide", side_effect=ProviderError("key rejected (401)", fatal=True)):
            code, _ = self.run_agent(server)
        self.assertEqual(code, 1)
        self.assertEqual(len(server.calls("POST", LOG)), 1)  # still explained in the run log

    def test_choosing_none_is_logged_too(self):
        server = self.serve()
        with mock.patch("agent.providers.Rules.decide", return_value={"action": "none", "reasoning": "Too choppy."}):
            code, _ = self.run_agent(server)
        self.assertEqual(code, 0)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        self.assertEqual(server.calls("POST", LOG)[0]["body"]["content"],
                         "No trade today. Too choppy. Context: market open · SPY +0.42% · QQQ +1.21% · DIA -0.03%.")

    def test_a_public_section_that_fails_is_skipped(self):
        server = self.serve(sectors=(503, {"detail": "busy"}))
        code, output = self.run_agent(server, "--dry-run")
        self.assertEqual(code, 0, output)
        self.assertIn("Warning: skipped sector sentiment", output)

    def test_github_step_summary(self):
        server = self.serve()
        summary = self.logs / "summary.md"
        code, _ = self.run_agent(server, GITHUB_STEP_SUMMARY=str(summary), GITHUB_ACTIONS="true")
        self.assertEqual(code, 0)
        text = summary.read_text()
        self.assertIn("### Alpha Bazaar agent: fixture_agent", text)
        self.assertIn("BUY NIC", text)


if __name__ == "__main__":
    unittest.main()
