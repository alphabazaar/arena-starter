"""The ai-hedge-fund adapter: v2 session records and v1 decisions, mapped to Alpha Bazaar and published to a fake
API. No network, no keys, and ai-hedge-fund isn't installed: --run gets stand-in modules."""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from adapters.ai_hedge_fund import adapter as aihf
from adapters.common import EXIT_LONG, EXIT_SHORT, FLAT, KEEP, LONG, SHORT, AdapterError
from tests.adapters_support import CLOSE_HELD, LOG, ME, SUBMIT, AdapterCase, closed_ok, would_send
from tests.fakes import fixture

RECORD = "aihf_session_record.json"  # what `aihf paper tick` prints and keeps in the ledger (2.7.2)
V1 = "aihf_v1_decisions.json"  # what v1's run_hedge_fund() returned


def parse(data):
    return aihf.parse(data, "test", None, "2026-11-12", aihf.DEFAULT_PORTFOLIO_VALUE)


def by_ticker(parsed):
    return {intent.ticker: intent for intent in parsed.intents}


def fake_hedge_fund(record, error=None):
    """Stand-ins for the parts of ai-hedge-fund the adapter imports for --run. error(NothingDue) is what tick raises."""
    calls = {"deploy": [], "tick": []}

    class NothingDue(RuntimeError):
        pass

    class FundHalted(RuntimeError):
        pass

    class FDClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Spec:
        name, is_intraday = "example-fund", False

    class Deployed:
        name, spec = "example-fund", Spec()

    class Record:
        def model_dump_json(self):
            return json.dumps(record)

    def deploy(name, spec, universe, root):
        calls["deploy"].append((name, list(universe)))
        directory = Path(root) / name
        directory.mkdir()
        return directory

    def tick(directory, data_client):
        calls["tick"].append(Path(directory))
        if error:
            raise error(NothingDue)
        return Record()

    def load_spec(path):
        calls["mandate"] = str(path)
        return Spec()

    names = ("hedge_fund", "hedge_fund.data", "hedge_fund.fund", "hedge_fund.paper", "hedge_fund.paths",
             "hedge_fund.pipeline", "hedge_fund.tui", "hedge_fund.tui.keys")
    modules = {name: types.ModuleType(name) for name in names}
    modules["hedge_fund"].__path__ = modules["hedge_fund.tui"].__path__ = []
    modules["hedge_fund.data"].FDClient = FDClient
    modules["hedge_fund.data"].CachedDataClient = lambda raw, live=False: raw
    modules["hedge_fund.fund"].load_spec = load_spec
    paper = modules["hedge_fund.paper"]
    paper.NothingDue, paper.deploy, paper.tick = NothingDue, deploy, tick
    paper.load_deployed, paper.validate_fund_name = lambda directory: Deployed(), lambda name: name
    modules["hedge_fund.paths"].EXAMPLE_MANDATE = Path("/site-packages/hedge_fund/fund/example.yaml")
    modules["hedge_fund.pipeline"].FundHalted = FundHalted
    modules["hedge_fund.tui.keys"].apply_credentials = lambda: calls.setdefault("credentials", True)
    return modules, calls


class FundRecords(unittest.TestCase):
    """v2 (2.x): target weights after risk limits, with each analyst's conviction."""

    def test_target_weights_map_to_what_alpha_bazaar_holds(self):
        parsed = parse(fixture(RECORD))
        intents = by_ticker(parsed)
        self.assertEqual({t: i.target for t, i in intents.items()},
                         {"AAPL": SHORT, "MSFT": FLAT, "NVDA": LONG, "TSLA": KEEP})  # TSLA was skipped
        self.assertEqual((parsed.source, parsed.model, parsed.fund),
                         ("ai-hedge-fund 2.7.2 (claude-opus-5-5)", "claude-opus-5-5", "alpha"))
        self.assertFalse(intents["MSFT"].minor)  # the fund holds MSFT, so a zero weight is an exit

    def test_confidence_and_size_come_from_the_fund(self):
        intents = by_ticker(parse(fixture(RECORD)))
        # net conviction: slice-weighted views, NVDA 0.6 * 0.55 + 0.4 * 0.3 = 0.45, AAPL 0.6 * 0.1 - 0.4 * 0.5 = -0.14
        self.assertEqual((intents["NVDA"].confidence, intents["AAPL"].confidence, intents["MSFT"].confidence),
                         (5, 1, None))
        self.assertEqual((intents["NVDA"].size_hint, intents["AAPL"].size_hint), (12.5, 5.0))
        self.assertIn("graham +0.60: Earnings power has compounded", intents["NVDA"].summary)

    def test_a_session_without_a_decision_publishes_nothing_new(self):
        parsed = parse({**fixture(RECORD), "decision": None, "rebalance": False})
        self.assertEqual(parsed.intents, [])
        self.assertIn("No rebalance at the 2026-11-11 session, so there's no new decision to publish. The fund holds "
                      "MSFT +22.", parsed.notes)

    def test_the_decision_a_session_executed_is_used_when_it_made_none(self):
        record = fixture(RECORD)
        record.update(executed=record["decision"], decision=None)
        parsed = parse(record)
        self.assertEqual(by_ticker(parsed)["NVDA"].target, LONG)
        self.assertIn("it executed the one made at the 2026-11-11 close", parsed.notes[0])

    def test_a_drawdown_stop_out_closes_everything(self):
        record = fixture(RECORD)
        record["decision"].update(final_weights={"MSFT": 0.0}, strategies=[], skipped=[],
                                  clamps=[{"limit": "max_drawdown_pct", "before": 0.21, "after": 0.0}])
        parsed = parse(record)
        self.assertEqual({i.target for i in parsed.intents}, {FLAT})
        self.assertIn("Drawdown stop-out", parsed.notes[0])

    def test_a_backtest_publishes_its_last_session(self):
        first = fixture(RECORD)
        first["decision"]["final_weights"] = {"AAPL": 0.2}
        self.assertEqual(by_ticker(parse({"records": [first, fixture(RECORD)]}))["NVDA"].target, LONG)


class V1Decisions(unittest.TestCase):
    """v1 (until 2026.7.10): an action per ticker, with shares, confidence 0-100 and reasoning."""

    def test_each_action_maps_conservatively(self):
        intents = by_ticker(parse(fixture(V1)))
        self.assertEqual({t: i.target for t, i in intents.items()},
                         {"AAPL": LONG, "MSFT": KEEP, "TSLA": SHORT, "KGC": EXIT_LONG, "AMD": KEEP})  # AMD: 0 shares
        cover = parse({"NVDA": {"action": "cover", "quantity": 5, "confidence": 50, "reasoning": "r"}}).intents[0]
        self.assertEqual(cover.target, EXIT_SHORT)
        self.assertEqual(intents["AMD"].why, "0 shares is no trade")

    def test_confidence_and_sizing_hints(self):
        intents = by_ticker(parse(fixture(V1)))
        self.assertEqual((intents["AAPL"].confidence, intents["TSLA"].confidence), (9, 7))  # 85 and 70 out of 100
        self.assertAlmostEqual(intents["AAPL"].size_hint, 43 * 232.1 / 100_000 * 100)  # shares x price / portfolio
        self.assertIn("Analysts: 2 bullish, 1 neutral.", intents["AAPL"].summary)

    def test_other_json_is_refused(self):
        for data in ({"foo": 1}, [], {"records": []}, {"decisions": None}):
            with self.subTest(data=data), self.assertRaises(AdapterError):
                parse(data)


class Publishing(AdapterCase):
    entry = staticmethod(aihf.main)

    def picks_with(self, *extra):
        picks = fixture("my_picks.json")
        picks["picks"] += list(extra)
        return 200, picks

    def test_a_fund_session_opens_longs_and_shorts_and_closes_exits(self):
        msft = {"id": "pick-msft", "ticker": "MSFT", "action": "buy", "status": "open", "position_size": 5.0,
                "created_at": "2026-11-04T15:00:00Z"}
        server = self.serve(picks=self.picks_with(msft))
        server.routes[("POST", "/api/performance/close-signal/pick-msft")] = closed_ok(514.0)
        code, output = self.run_adapter(server, "--file", self.write(RECORD))
        self.assertEqual(code, 0, output)
        # Exits first, then new signals by conviction; KGC isn't in the fund's universe, so it's left alone
        posts = [(r["path"], (r["body"] or {}).get("ticker")) for r in server.requests if r["method"] == "POST"]
        self.assertEqual(posts,
                         [("/api/performance/close-signal/pick-msft", None), (SUBMIT, "NVDA"), (SUBMIT, "AAPL"),
                          (LOG, None)])
        nvda, aapl = (r["body"] for r in server.calls("POST", SUBMIT))
        self.assertEqual((nvda["action"], nvda["confidence"], nvda["position_size"], nvda["time_horizon"],
                          nvda["auto_close_days"]), ("buy", 5, 5.0, "1M", 30))
        self.assertEqual((aapl["action"], aapl["confidence"]), ("short", 1))
        self.assertTrue(nvda["reasoning"].startswith("ai-hedge-fund 2.7.2 (claude-opus-5-5): fund alpha targets NVDA "
                                                     "at +12.5% of NAV after the 2026-11-11 close."))
        self.assertEqual(server.calls("PATCH", ME)[0]["body"], {"framework": "ai-hedge-fund",
                                                                "model": "claude-opus-5-5"})
        content = self.log_entry(server)["content"]
        self.assertIn("MSFT: target +0.0% of NAV, holds +22 shares → closed the open long (opened 2026-11-04).",
                      content)
        self.assertIn("TSLA: was skipped by the fund → no trade: ai-hedge-fund skipped it", content)
        self.assertNotIn("KGC", content)
        self.assert_no_key_anywhere(output)

    def test_size_from_fund_uses_its_weights(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--file", self.write(RECORD), "--size-from-fund")
        self.assertEqual(code, 0, output)
        self.assertEqual({r["body"]["ticker"]: r["body"]["position_size"] for r in server.calls("POST", SUBMIT)},
                         {"NVDA": 12.5, "AAPL": 5.0})

    def test_v1_dry_run_shows_every_request_and_sends_none(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--file", self.write(V1), "--dry-run")
        self.assertEqual(code, 0, output)
        self.assertEqual({r["method"] for r in server.requests}, {"GET"})
        shown = would_send(output)
        self.assertIn(("POST", CLOSE_HELD[1], None), shown)  # sell closes the open KGC long
        self.assertEqual([(b["ticker"], b["action"], b["confidence"], b["position_size"])
                          for method, path, b in shown if path == SUBMIT],
                         [("AAPL", "buy", 9, 5.0), ("TSLA", "short", 7, 5.0)])
        [entry] = [b for method, path, b in shown if path == LOG]
        self.assertIn("MSFT: decided hold 0 shares, confidence 55/100 → no trade: hold keeps what's there.",
                      entry["content"])
        self.assertIn(("PATCH", ME, {"framework": "ai-hedge-fund"}), shown)  # v1 names no model: that label stays

    def test_a_rejected_decision_is_a_run_log_note_and_the_rest_still_go(self):
        server = self.serve()
        accept = server.routes[("POST", SUBMIT)]
        server.routes[("POST", SUBMIT)] = lambda request: (
            (400, {"detail": "Ticker 'TSLA' is not supported for short signals"})
            if request["body"]["ticker"] == "TSLA" else accept(request))
        server.routes[CLOSE_HELD] = closed_ok()
        code, output = self.run_adapter(server, "--file", self.write(V1))
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", output)
        self.assertEqual([r["body"]["ticker"] for r in server.calls("POST", SUBMIT)], ["AAPL", "TSLA"])
        entry = self.log_entry(server)
        self.assertIn("TSLA: decided short 12 shares, confidence 70/100 → tried to publish a short, but Alpha Bazaar "
                      "rejected it: Ticker 'TSLA' is not supported for short signals.", entry["content"])
        self.assertIn("KGC: decided sell 100 shares, confidence 65/100 → closed the open long (opened 2026-10-15).",
                      entry["content"])
        self.assertEqual((entry["meta"]["published"], entry["meta"]["closed"], entry["meta"]["rejected"]), (1, 1, 1))

    def test_one_decision_from_the_command_line(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--ticker", "aapl", "--action", "BUY", "--quantity", "10",
                                        "--confidence", "72", "--reasoning", "Cheap on cash flow.")
        self.assertEqual(code, 0, output)
        body = server.calls("POST", SUBMIT)[0]["body"]
        self.assertEqual((body["ticker"], body["action"], body["confidence"]), ("AAPL", "buy", 7))
        self.assertEqual(body["reasoning"], "ai-hedge-fund decided to buy 10 shares of AAPL with confidence 72/100. "
                                            "Cheap on cash flow.")
        self.assertEqual(server.calls("PATCH", ME)[0]["body"], {"framework": "ai-hedge-fund"})  # model left alone

    def test_a_full_portfolio_means_no_room(self):
        full = {"status": "success", "has_more": False, "picks": [
            {"id": f"p{i}", "ticker": f"T{i}", "action": "buy", "status": "open", "position_size": 10.0}
            for i in range(10)]}
        server = self.serve(picks=(200, full))
        code, output = self.run_adapter(server, "--ticker", "AAPL", "--action", "buy", "--quantity", "10")
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        self.assertIn("AAPL: decided buy 10 shares → no trade: wanted a long, but the paper portfolio is fully "
                      "allocated", self.log_entry(server)["content"])

    def test_a_paper_fund_publishes_its_newest_ledger_record(self):
        ledger = self.tmp / "paper" / "alpha" / "ledger"
        (ledger / "superseded").mkdir(parents=True)
        older = fixture(RECORD)
        older.update(session="2026-11-10", decision={**older["decision"], "final_weights": {"MSFT": 0.3}})
        (ledger / "2026-11-10.json").write_text(json.dumps(older))
        (ledger / "2026-11-11.json").write_text(json.dumps(fixture(RECORD)))
        (ledger / "superseded" / "2026-11-12.json").write_text("{}")
        server = self.serve()
        code, output = self.run_adapter(server, "--fund", "alpha", "--paper-dir", str(self.tmp / "paper"))
        self.assertEqual(code, 0, output)
        self.assertEqual([r["body"]["ticker"] for r in server.calls("POST", SUBMIT)], ["NVDA", "AAPL"])

    def test_piped_from_aihf_paper_tick(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--file", "-", stdin=json.dumps(fixture(RECORD)))
        self.assertEqual(code, 0, output)
        self.assertEqual(len(server.calls("POST", SUBMIT)), 2)

    def test_no_rebalance_is_still_explained(self):
        server = self.serve()
        record = {**fixture(RECORD), "decision": None, "rebalance": False, "session": "2026-11-12"}
        code, output = self.run_adapter(server, "--file", self.write(record))
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        entry = self.log_entry(server)
        self.assertIn("No rebalance at the 2026-11-12 session", entry["content"])
        self.assertIsNone(entry["ticker"])

    def test_run_mode_ticks_a_fresh_fund_then_publishes(self):
        modules, calls = fake_hedge_fund(fixture(RECORD))
        server = self.serve()
        with mock.patch.dict(sys.modules, modules):
            code, output = self.run_adapter(server, "--run", "--universe", "aapl,msft nvda,TSLA")
        self.assertEqual(code, 0, output)
        self.assertEqual(calls["deploy"], [("example-fund", ["AAPL", "MSFT", "NVDA", "TSLA"])])
        self.assertEqual(calls["mandate"], "/site-packages/hedge_fund/fund/example.yaml")  # its example mandate
        self.assertTrue(calls["credentials"])
        self.assertEqual(len(server.calls("POST", SUBMIT)), 2)

    def test_a_failed_tick_fails_the_run_and_publishes_nothing(self):
        modules, calls = fake_hedge_fund(fixture(RECORD),
                                         error=lambda NothingDue: RuntimeError("FINANCIAL_DATASETS_API_KEY is not set"))
        server = self.serve()
        with mock.patch.dict(sys.modules, modules):
            code, output = self.run_adapter(server, "--run", "--universe", "NVDA")
        self.assertEqual(code, 1)
        self.assertIn("ai-hedge-fund's tick failed: RuntimeError: FINANCIAL_DATASETS_API_KEY is not set", output)
        self.assertEqual([r for r in server.requests if r["method"] != "GET"], [])

    def test_nothing_due_is_not_an_error(self):
        modules, calls = fake_hedge_fund(fixture(RECORD), error=lambda NothingDue: NothingDue(
            "alpha: no completed SPY session after 2026-11-11"))
        server = self.serve()
        with mock.patch.dict(sys.modules, modules):
            code, output = self.run_adapter(server, "--run", "--fund", "alpha", "--paper-dir", str(self.tmp))
        self.assertEqual(code, 2, output)  # no fund.yaml there yet: the deployed fund must exist
        (self.tmp / "alpha").mkdir()
        (self.tmp / "alpha" / "fund.yaml").write_text("name: alpha\n")
        with mock.patch.dict(sys.modules, modules):
            code, output = self.run_adapter(server, "--run", "--fund", "alpha", "--paper-dir", str(self.tmp))
        self.assertEqual(code, 0, output)
        self.assertEqual(calls["tick"], [self.tmp / "alpha"])
        self.assertIn("no completed SPY session after 2026-11-11", output)
        self.assertEqual([r for r in server.requests if r["method"] != "GET"], [])

    def test_run_mode_without_ai_hedge_fund_says_how_to_install_it(self):
        server = self.serve()
        with mock.patch.dict(sys.modules, {"hedge_fund": None}):
            code, output = self.run_adapter(server, "--run", "--universe", "NVDA")
        self.assertEqual(code, 2)
        self.assertIn("pip install aihf==2.7.2", output)

    def test_setup_mistakes_send_nothing(self):
        server = self.serve()
        for args in ([], ["--file", "x.json", "--action", "buy"], ["--universe", "NVDA"], ["--action", "buy"],
                     ["--ticker", "NVDA", "--action", "maybe"], ["--ticker", "NVDA", "--quantity", "3"],
                     ["--run"], ["--file", str(self.tmp / "missing.json")]):
            with self.subTest(args=args):
                code, output = self.run_adapter(server, *args)
                self.assertEqual(code, 2, output)
        self.assertEqual([r for r in server.requests if r["method"] != "GET"], [])



if __name__ == "__main__":
    unittest.main()
