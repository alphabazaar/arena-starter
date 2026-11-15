"""The TradingAgents adapter: reading its output, mapping ratings conservatively, and publishing to a fake Alpha
Bazaar API. No network, no keys, and TradingAgents isn't installed: --run gets a stand-in module."""
import json
import sys
import types
import unittest
from unittest import mock

from adapters.common import EXIT_LONG, KEEP, LONG, SHORT, AdapterError
from adapters.tradingagents import adapter as ta
from tests.adapters_support import (CLOSE_HELD, LOG, ME, SUBMIT, AdapterCase, close_queued, closed_ok,
                                    picks_by_status, would_send)
from tests.fakes import TEST_KEY, fixture

STATE = "tradingagents_state_log.json"  # what TradingAgents v0.7.1 saves as full_states_log_<date>.json
SOURCE = "TradingAgents 0.7.1 (gpt-6-sol)"


def state(**changes):
    return {**fixture(STATE), **changes}


def fake_tradingagents(propagate):
    """Stand-ins for the parts of TradingAgents the adapter imports: TradingAgentsGraph and DEFAULT_CONFIG."""
    package = types.ModuleType("tradingagents")
    package.__version__, package.__path__ = "0.7.1", []
    default_config = types.ModuleType("tradingagents.default_config")
    default_config.DEFAULT_CONFIG = {"llm_provider": "openai", "deep_think_llm": "gpt-6-sol"}
    graph = types.ModuleType("tradingagents.graph")
    graph.__path__ = []
    trading_graph = types.ModuleType("tradingagents.graph.trading_graph")

    class TradingAgentsGraph:
        instances = []

        def __init__(self, selected_analysts=("market", "social", "news", "fundamentals"), debug=False, config=None,
                     callbacks=None):
            self.selected_analysts, self.debug, self.config = selected_analysts, debug, config
            TradingAgentsGraph.instances.append(self)

        def run_settings(self):
            return {"version": "0.7.1", "deep_think_llm": "gpt-6-sol", "quick_think_llm": "gpt-6-luna"}

        def propagate(self, company_name, trade_date):
            return propagate(company_name, trade_date)

    trading_graph.TradingAgentsGraph = TradingAgentsGraph
    modules = {"tradingagents": package, "tradingagents.default_config": default_config,
               "tradingagents.graph": graph, "tradingagents.graph.trading_graph": trading_graph}
    return modules, TradingAgentsGraph


class Parsing(unittest.TestCase):
    def test_a_saved_run(self):
        [decision] = ta.parse(fixture(STATE), "fixture")
        self.assertEqual((decision.ticker, decision.rating, decision.as_of, decision.model, decision.version),
                         ("NVDA", "Buy", "2026-11-12", "gpt-6-sol", "0.7.1"))
        self.assertTrue(decision.summary.startswith("Enter a standard 5% position near $186"))  # the executive summary
        self.assertEqual(decision.levels, {"price_target": 212.0, "stop_loss": 171.5})

    def test_the_rating_is_read_from_the_decision_when_it_is_missing(self):
        saved = state(final_trade_decision=fixture(STATE)["final_trade_decision"].replace("Buy", "Underweight"))
        del saved["final_rating"]
        self.assertEqual(ta.parse(saved, "x")[0].rating, "Underweight")

    def test_older_versions(self):
        self.assertEqual([ta.normalize_rating(v) for v in ("BUY", "hold", "Sell", "**Overweight**", "maybe")],
                         ["Buy", "Hold", "Sell", "Overweight", None])
        self.assertEqual(ta.rating_from_text("The plan.\nFINAL TRANSACTION PROPOSAL: **SELL**"), "Sell")
        self.assertIsNone(ta.rating_from_text("Our view: this is not a Sell yet."))
        saved = state()
        del saved["final_rating"], saved["run_settings"]
        [decision] = ta.parse({"2026-11-12": saved}, "x")  # v0.1 kept the state under its date
        self.assertEqual((decision.rating, decision.version, decision.model), ("Buy", None, None))

    def test_no_readable_rating_is_review(self):
        self.assertEqual(ta.parse(state(final_rating="REVIEW", final_trade_decision="No clear call."), "x")[0].rating,
                         ta.REVIEW)

    def test_small_decision_objects(self):
        first = {"ticker": "$nvda", "rating": "buy", "reasoning": "r", "date": "2026-11-12"}
        decisions = ta.parse({"decisions": [first, {"ticker": "AAPL", "signal": "HOLD"}]}, "x")
        self.assertEqual([(d.ticker, d.rating, d.as_of) for d in decisions],
                         [("NVDA", "Buy", "2026-11-12"), ("AAPL", "Hold", None)])

    def test_other_json_is_refused(self):
        for data in ({"foo": 1}, [1, 2], "text", {"ticker": "NVDA"}):
            with self.subTest(data=data), self.assertRaises(AdapterError):
                ta.parse(data, "x")


class Mapping(unittest.TestCase):
    def intent(self, rating, **options):
        return ta.to_intent(ta.Decision(ticker="NVDA", rating=rating, as_of="2026-11-12"), SOURCE, **options)

    def test_ratings_map_conservatively(self):
        self.assertEqual({rating: self.intent(rating).target for rating in (*ta.RATINGS, ta.REVIEW)},
                         {"Buy": LONG, "Overweight": LONG, "Hold": KEEP, "Underweight": EXIT_LONG, "Sell": EXIT_LONG,
                          "REVIEW": KEEP})

    def test_sell_is_a_short_only_when_asked(self):
        self.assertEqual(self.intent("Sell", sell_as_short=True).target, SHORT)
        self.assertEqual(self.intent("Underweight", sell_as_short=True).target, EXIT_LONG)

    def test_nothing_is_invented(self):
        intent = self.intent("Buy")
        self.assertIsNone(intent.confidence)  # TradingAgents gives no conviction number: Alpha Bazaar's default
        self.assertEqual(intent.headline, "TradingAgents 0.7.1 (gpt-6-sol) rated NVDA Buy in its 2026-11-12 analysis.")


class Publishing(AdapterCase):
    entry = staticmethod(ta.main)

    def test_buy_publishes_a_long_and_explains_it(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--file", self.write(STATE))
        self.assertEqual(code, 0, output)

        [submit] = server.calls("POST", SUBMIT)
        body = submit["body"]
        self.assertEqual({k: body[k] for k in ("ticker", "action", "time_horizon", "position_size", "auto_close_days")},
                         {"ticker": "NVDA", "action": "buy", "time_horizon": "1W", "position_size": 5.0,
                          "auto_close_days": 7})
        self.assertNotIn("confidence", body)
        self.assertNotIn("stop_loss", body)  # levels are opt-in
        self.assertTrue(body["reasoning"].startswith(
            "TradingAgents 0.7.1 (gpt-6-sol) rated NVDA Buy in its 2026-11-12 analysis. Enter a standard 5% position"))
        self.assertIn("adapter/tradingagents", submit["headers"]["user-agent"])

        # The profile says what decides, and the run log says what was done and why
        self.assertEqual(server.calls("PATCH", ME)[0]["body"], {"framework": "TradingAgents", "model": "gpt-6-sol"})
        entry = self.log_entry(server)
        self.assertEqual((entry["kind"], entry["ticker"]), ("decision", "NVDA"))
        self.assertIn("NVDA: rated Buy → published a long, 5% of the paper portfolio for up to 7 days (1W). Why: Enter "
                      "a standard 5% position", entry["content"])
        self.assertEqual({k: entry["meta"][k] for k in ("framework", "framework_version", "published", "calls")},
                         {"framework": "TradingAgents", "framework_version": "0.7.1", "published": 1,
                          "calls": {"NVDA": "Buy"}})
        self.assertLessEqual(len(json.dumps(entry["meta"]).encode()), 2048)
        self.assertIn("Attestation for NEW: AB-new123", output)
        self.assert_no_key_anywhere(output)

    def test_a_dry_run_prints_exactly_what_a_live_run_sends(self):
        dry = self.serve()
        code, output = self.run_adapter(dry, "--file", self.write(STATE), "--dry-run")
        self.assertEqual(code, 0, output)
        self.assertEqual([r["method"] for r in dry.requests], ["GET"] * 3)  # profile, open and recent signals only
        self.assertIn("Dry run: nothing was sent", output)
        shown = {(method, path): body for method, path, body in would_send(output)}

        live = self.serve()
        code, _ = self.run_adapter(live, "--file", self.write(STATE))
        self.assertEqual(code, 0)
        sent = {(r["method"], r["path"]): r["body"] for r in live.requests if r["method"] != "GET"}
        self.assertEqual(shown, sent)

    def test_a_dry_run_without_a_key_makes_no_requests(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--ticker", "NVDA", "--rating", "Buy", "--dry-run", key=None)
        self.assertEqual(code, 0, output)
        self.assertEqual(server.requests, [])
        self.assertIn("Would send: POST /api/publisher/submit-pick", output)

    def test_hold_is_a_logged_no_trade(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--ticker", "NVDA", "--rating", "hold", "--reasoning", "Balanced.")
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        self.assertIn("NVDA: rated Hold → no trade: Hold keeps what's there. Why: Balanced.",
                      self.log_entry(server)["content"])

    def test_sell_closes_the_open_long(self):
        server = self.serve()
        server.routes[CLOSE_HELD] = closed_ok()
        code, output = self.run_adapter(server, "--ticker", "KGC", "--rating", "Sell")
        self.assertEqual(code, 0, output)
        self.assertEqual(len(server.calls(*CLOSE_HELD)), 1)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        entry = self.log_entry(server)
        self.assertIn("KGC: rated Sell → closed the open long (opened 2026-10-15).", entry["content"])
        self.assertEqual(entry["meta"]["closed"], 1)

    def test_underweight_closes_the_open_long_too(self):
        server = self.serve()
        server.routes[CLOSE_HELD] = closed_ok()
        code, output = self.run_adapter(server, "--ticker", "KGC", "--rating", "Underweight")
        self.assertEqual(code, 0, output)
        self.assertEqual(len(server.calls(*CLOSE_HELD)), 1)

    def test_sell_without_an_open_long_is_no_trade_not_a_short(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--ticker", "NVDA", "--rating", "Sell")
        self.assertEqual(code, 0, output)
        self.assertEqual([r for r in server.requests if r["path"] in (SUBMIT, CLOSE_HELD[1])], [])
        self.assertIn("NVDA: rated Sell → no trade: Sell means exit or avoid, and there's no open long to exit "
                      "(publishing Sell as a short takes --sell-as-short).", self.log_entry(server)["content"])

    def test_sell_as_short_closes_the_long_then_shorts(self):
        server = self.serve()
        server.routes[CLOSE_HELD] = closed_ok()
        code, output = self.run_adapter(server, "--ticker", "KGC", "--rating", "Sell", "--sell-as-short")
        self.assertEqual(code, 0, output)
        self.assertEqual([r["path"] for r in server.requests if r["method"] == "POST"], [CLOSE_HELD[1], SUBMIT, LOG])
        self.assertEqual(server.calls("POST", SUBMIT)[0]["body"]["action"], "short")

    def test_after_hours_a_reversal_waits_for_the_queued_close(self):
        server = self.serve()
        server.routes[CLOSE_HELD] = close_queued()
        code, output = self.run_adapter(server, "--ticker", "KGC", "--rating", "Sell", "--sell-as-short")
        self.assertEqual(code, 0, output)
        self.assertEqual(len(server.calls(*CLOSE_HELD)), 1)
        self.assertEqual(server.calls("POST", SUBMIT), [])  # no short while the long is still open overnight
        entry = self.log_entry(server)
        self.assertIn("KGC: rated Sell (published as a short: --sell-as-short) → close of the open long (opened "
                      "2026-10-15) queued for the next open; run again after it to publish the short.", entry["content"])
        self.assertNotIn("closed the open long", entry["content"])
        self.assertEqual((entry["meta"]["closed"], entry["meta"]["close_queued"]), (0, 1))

    def test_rerunning_the_same_output_does_not_publish_it_again(self):
        open_picks = fixture("my_picks.json")
        for side, expected in (("buy", 0), ("short", 1)):  # a closed long from that day counts; a short doesn't
            with self.subTest(side=side):
                recent = fixture("my_picks.json")
                recent["picks"].append({"id": "pick-nvda", "ticker": "NVDA", "action": side, "status": "closed",
                                        "position_size": 5.0, "created_at": "2026-11-12T15:31:00Z"})
                server = self.serve(picks=picks_by_status(open_picks, recent))
                code, output = self.run_adapter(server, "--file", self.write(STATE))
                self.assertEqual(code, 0, output)
                self.assertEqual(len(server.calls("POST", SUBMIT)), expected)
                if not expected:
                    self.assertIn("NVDA: rated Buy → no trade: a long was already published on 2026-11-12, on or "
                                  "after this decision's date (2026-11-12), so it isn't published again.",
                                  self.log_entry(server)["content"])

    def test_buy_keeps_the_open_long(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--ticker", "KGC", "--rating", "Buy")
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        self.assertIn("KGC: rated Buy → no trade: the open long from 2026-10-15 stays open.",
                      self.log_entry(server)["content"])

    def test_a_rejected_signal_is_a_run_log_note_not_a_crash(self):
        server = self.serve(submit=(400, {"detail": "Ticker 'ZZZZ' not found"}))
        code, output = self.run_adapter(server, "--ticker", "ZZZZ", "--rating", "Buy")
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", output)
        entry = self.log_entry(server)
        self.assertIn("ZZZZ: rated Buy → tried to publish a long, but Alpha Bazaar rejected it: Ticker 'ZZZZ' not "
                      "found.", entry["content"])
        self.assertEqual(entry["meta"]["rejected"], 1)

    def test_the_weekly_limit_stops_the_signals_after_it(self):
        server = self.serve(submit=(429, {"detail": "Weekly signal limit reached (50 signals/week)"}))
        path = self.write([{"ticker": "NVDA", "rating": "Buy", "date": "2026-11-12"},
                           {"ticker": "AAPL", "rating": "Overweight", "date": "2026-11-12"}])
        code, output = self.run_adapter(server, "--file", path)
        self.assertEqual(code, 1)
        self.assertEqual(len(server.calls("POST", SUBMIT)), 1)  # AAPL was never sent
        content = self.log_entry(server)["content"]
        self.assertIn("NVDA: rated Buy → tried to publish a long, but Alpha Bazaar rejected it: Weekly signal limit "
                      "reached (50 signals/week).", content)
        self.assertIn("AAPL: rated Overweight → wanted a long, but Alpha Bazaar's limit stopped the signals before "
                      "it", content)

    def test_a_refused_close_stops_the_reversal(self):
        server = self.serve()
        server.routes[CLOSE_HELD] = (409, {"detail": "Signal already closed"})
        code, output = self.run_adapter(server, "--ticker", "KGC", "--rating", "Sell", "--sell-as-short")
        self.assertEqual(code, 1)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        self.assertIn("KGC: rated Sell (published as a short: --sell-as-short) → tried to close the open long (opened "
                      "2026-10-15), but Alpha Bazaar answered: Signal already closed; so no short was published (one "
                      "position per ticker).", self.log_entry(server)["content"])

    def test_an_old_decision_is_not_published_as_a_new_call(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--file", self.write(state(trade_date="2026-10-01")))
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT), [])
        self.assertIn("NVDA: rated Buy → no trade: it was decided for 2026-10-01, 42 days ago, too long ago to publish "
                      "as a new call", self.log_entry(server)["content"])

    def test_a_run_log_failure_fails_the_run(self):
        server = self.serve(log=(422, {"detail": [{"loc": ["body", "content"], "msg": "too long"}]}))
        code, output = self.run_adapter(server, "--ticker", "NVDA", "--rating", "Hold")
        self.assertEqual(code, 1)
        self.assertIn("couldn't write the run log", output)

    def test_keys_and_links_never_reach_alpha_bazaar(self):
        server = self.serve()
        secret = "sk-proj-abcdefghijklmnop1234"
        code, output = self.run_adapter(server, "--ticker", "NVDA", "--rating", "Buy", "--reasoning",
                                        f"Per https://news.example.com/a?token={secret} and {TEST_KEY}, buy.")
        self.assertEqual(code, 0, output)
        reasoning = server.calls("POST", SUBMIT)[0]["body"]["reasoning"]
        for text in (reasoning, self.log_entry(server)["content"], output):
            for leak in (secret, TEST_KEY, "token="):
                self.assertNotIn(leak, text)
        self.assertIn("Per news.example.com and ab_live_***, buy.", reasoning)

    def test_levels_are_opt_in_and_dropped_when_the_live_price_moved(self):
        server = self.serve()
        accept, sent = server.routes[("POST", SUBMIT)], []

        def submit(request):  # Alpha Bazaar's live price ($168) is under the Trader's stop: refused once
            sent.append(request["body"])
            return (400, {"detail": "stop_loss must be below current price ($168.00) for buy signals. Got $171.50."}
                    ) if len(sent) == 1 else accept(request)

        server.routes[("POST", SUBMIT)] = submit
        code, output = self.run_adapter(server, "--file", self.write(STATE), "--levels")
        self.assertEqual(code, 0, output)
        self.assertEqual((sent[0]["stop_loss"], sent[0]["price_target"]), (171.5, 212.0))
        self.assertFalse({"stop_loss", "price_target"} & set(sent[1]))

    def test_latest_publishes_the_newest_saved_run(self):
        folder = self.tmp / "results" / "NVDA" / "TradingAgentsStrategy_logs"
        folder.mkdir(parents=True)
        for day, rating in (("2026-11-10", "Sell"), ("2026-11-12", "Buy"), ("2026-11-11", "Hold")):
            (folder / f"full_states_log_{day}.json").write_text(json.dumps(state(trade_date=day, final_rating=rating)))
        server = self.serve()
        code, output = self.run_adapter(server, "--latest", "nvda,AAPL", "--results-dir", str(self.tmp / "results"))
        self.assertEqual(code, 0, output)
        self.assertEqual(server.calls("POST", SUBMIT)[0]["body"]["ticker"], "NVDA")
        self.assertIn("no saved TradingAgents run for AAPL", output)

    def test_run_mode_runs_tradingagents_then_publishes(self):
        seen = []

        def propagate(ticker, day):
            seen.append((ticker, day))
            return state(company_of_interest=ticker, trade_date=day, final_rating="Overweight"), "Overweight"

        modules, graph = fake_tradingagents(propagate)
        server = self.serve()
        with mock.patch.dict(sys.modules, modules):
            code, output = self.run_adapter(server, "--run", "nvda", "--analysts", "market,sentiment")
        self.assertEqual(code, 0, output)
        self.assertEqual(seen, [("NVDA", "2026-11-12")])  # today in New York
        self.assertEqual(graph.instances[0].selected_analysts, ["market", "social"])
        self.assertEqual(server.calls("POST", SUBMIT)[0]["body"]["action"], "buy")
        self.assertIn("NVDA: rated Overweight → published a long", self.log_entry(server)["content"])

    def test_a_ticker_tradingagents_fails_on_is_logged_and_flagged(self):
        def propagate(ticker, day):
            if ticker == "AAPL":
                raise RuntimeError("no price data")
            return state(trade_date=day, final_rating="Hold"), "Hold"

        modules, _ = fake_tradingagents(propagate)
        server = self.serve()
        with mock.patch.dict(sys.modules, modules):
            code, output = self.run_adapter(server, "--run", "NVDA,AAPL")
        self.assertEqual(code, 1)
        content = self.log_entry(server)["content"]
        self.assertIn("NVDA: rated Hold → no trade", content)
        self.assertIn("AAPL: couldn't be analyzed → no trade: TradingAgents didn't finish (RuntimeError).", content)

    def test_run_mode_without_tradingagents_says_how_to_install_it(self):
        server = self.serve()
        with mock.patch.dict(sys.modules, {"tradingagents": None}):
            code, output = self.run_adapter(server, "--run", "NVDA")
        self.assertEqual(code, 2)
        self.assertIn('pip install "tradingagents @ git+https://github.com/TauricResearch/TradingAgents@v0.7.1"',
                      output)
        self.assertEqual([r["path"] for r in server.requests], [ME])  # only the key check

    def test_a_bad_key_stops_before_tradingagents_spends_anything(self):
        modules, graph = fake_tradingagents(lambda ticker, day: self.fail("TradingAgents ran"))
        server = self.serve(me=(401, {"detail": "Invalid or revoked API key"}))
        with mock.patch.dict(sys.modules, modules):
            code, output = self.run_adapter(server, "--run", "NVDA")
        self.assertEqual(code, 1)
        self.assertIn("Check AB_API_KEY", output)
        self.assertEqual(graph.instances, [])

    def test_a_live_run_needs_a_key(self):
        server = self.serve()
        code, output = self.run_adapter(server, "--ticker", "NVDA", "--rating", "Buy", key=None)
        self.assertEqual(code, 2)
        self.assertIn("register.py", output)
        self.assertEqual(server.requests, [])

    def test_setup_mistakes_send_nothing(self):
        server = self.serve()
        for args in (["--rating", "Buy"], ["--ticker", "NVDA", "--rating", "Strong Buy"],
                     ["--latest", "NVDA", "--ticker", "NVDA"],
                     ["--ticker", "NVDA", "--rating", "Buy", "--horizon", "5Y"],
                     ["--ticker", "NVDA", "--rating", "Buy", "--size", "500"],
                     ["--file", str(self.tmp / "missing.json")],
                     ["--run", "NVDA", "--analysts", "astrology"]):
            with self.subTest(args=args):
                code, output = self.run_adapter(server, *args)
                self.assertEqual(code, 2, output)
        self.assertEqual([r for r in server.requests if r["method"] != "GET"], [])


if __name__ == "__main__":
    unittest.main()
