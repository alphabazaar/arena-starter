"""Decision validation, offline: what the model may answer, and what reaches Alpha Bazaar."""
import json
import unittest

from agent.decision import (Decision, InvalidDecision, REASONING_MAX, parse_json, size_position, validate)

CANDIDATES = {"NVDA": {"price": 100.0}, "BRK.B": {"price": 450.0}, "SPY": {"price": 766.48}}


def check(reply, held=frozenset(), **kwargs):
    return validate(reply, candidates=CANDIDATES, held=set(held), **kwargs)


def answer(**fields):
    base = {"action": "buy", "ticker": "NVDA", "time_horizon": "1W", "confidence": 7, "stop_loss_pct": 6,
            "take_profit_pct": 12, "reasoning": "Momentum and three sources agree."}
    base.update(fields)
    return json.dumps(base)


class ParseJson(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(parse_json('{"action": "none"}'), {"action": "none"})

    def test_code_fence_and_prose_around_it(self):
        reply = 'Here you go:\n```json\n{"action": "buy", "ticker": "NVDA"}\n```\nGood luck!'
        self.assertEqual(parse_json(reply)["ticker"], "NVDA")

    def test_skips_braces_that_are_not_json(self):
        self.assertEqual(parse_json('Think {step one} then {"action": "none"}'), {"action": "none"})

    def test_dict_passes_through(self):
        self.assertEqual(parse_json({"action": "none"}), {"action": "none"})

    def test_empty_or_no_object(self):
        for reply in ("", "   ", None, "no json here", "[1, 2]"):
            with self.subTest(reply=reply), self.assertRaises(InvalidDecision):
                parse_json(reply)


class Validate(unittest.TestCase):
    def test_full_buy(self):
        d = check(answer())
        self.assertEqual((d.action, d.ticker, d.time_horizon, d.confidence), ("buy", "NVDA", "1W", 7))
        self.assertEqual((d.stop_loss_pct, d.take_profit_pct, d.hold_days), (6, 12, 7))
        self.assertTrue(d.is_trade)

    def test_none_needs_only_reasoning(self):
        d = check('{"action": "none", "reasoning": "Nothing has an edge today."}')
        self.assertEqual(d.action, "none")
        self.assertFalse(d.is_trade)
        self.assertIsNone(d.ticker)

    def test_none_ignores_a_stray_ticker(self):
        self.assertIsNone(check(answer(action="none", ticker="ZZZZ")).ticker)

    def test_action_synonyms_and_case(self):
        self.assertEqual(check(answer(action="LONG")).action, "buy")
        self.assertEqual(check(answer(action="Hold")).action, "none")
        self.assertEqual(check(answer(action="short")).action, "short")

    def test_unknown_or_missing_action(self):
        for action in ("sell", "close", "", None, 1):
            with self.subTest(action=action), self.assertRaises(InvalidDecision):
                check(answer(action=action))

    def test_reasoning_is_required(self):
        for reasoning in ("", "   ", None, 42, "<b></b>"):
            with self.subTest(reasoning=reasoning), self.assertRaises(InvalidDecision):
                check(answer(reasoning=reasoning))

    def test_reasoning_is_cleaned_and_capped(self):
        d = check(answer(reasoning="  <b>Strong</b>\n\n setup  " + "word " * 1000))
        self.assertTrue(d.reasoning.startswith("Strong setup"))
        self.assertLessEqual(len(d.reasoning), REASONING_MAX)
        self.assertTrue(d.reasoning.endswith("…"))

    def test_ticker_rules(self):
        self.assertEqual(check(answer(ticker=" $nvda ")).ticker, "NVDA")
        self.assertEqual(check(answer(ticker="brk.b")).ticker, "BRK.B")
        for ticker in (None, "", "NV DA", "TOO_LONG_TICKER", "-ABC", 123):
            with self.subTest(ticker=ticker), self.assertRaises(InvalidDecision):
                check(answer(ticker=ticker))

    def test_only_candidates_unless_allowed(self):
        with self.assertRaisesRegex(InvalidDecision, "candidates"):
            check(answer(ticker="AAPL"))
        self.assertEqual(check(answer(ticker="AAPL"), allow_any_ticker=True).ticker, "AAPL")

    def test_no_second_position_in_a_held_ticker(self):
        with self.assertRaisesRegex(InvalidDecision, "already an open signal"):
            check(answer(), held={"NVDA"})
        with self.assertRaisesRegex(InvalidDecision, "already an open signal"):
            check(answer(action="short"), held={"NVDA"})

    def test_time_horizon(self):
        self.assertEqual(check(answer(time_horizon=None)).time_horizon, "1W")
        self.assertEqual(check(answer(time_horizon="1m")).time_horizon, "1M")
        self.assertEqual(check(answer(time_horizon="Swing")).hold_days, 10)
        for horizon in ("1 week", "5D", 7):
            with self.subTest(horizon=horizon), self.assertRaises(InvalidDecision):
                check(answer(time_horizon=horizon))

    def test_confidence(self):
        self.assertEqual(check(answer(confidence=None)).confidence, 5)
        self.assertEqual(check(answer(confidence="8")).confidence, 8)
        self.assertEqual(check(answer(confidence=9.0)).confidence, 9)
        for value in (0, 11, 0.8, 7.5, "high", True):
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                check(answer(confidence=value))

    def test_stop_and_target_ranges(self):
        d = check(answer(stop_loss_pct=None, take_profit_pct=0))
        self.assertIsNone(d.stop_loss_pct)
        self.assertIsNone(d.take_profit_pct)
        for field, value in (("stop_loss_pct", 0.1), ("stop_loss_pct", 60), ("stop_loss_pct", -5),
                             ("take_profit_pct", 250), ("take_profit_pct", "lots"), ("stop_loss_pct", float("nan"))):
            with self.subTest(field=field, value=value), self.assertRaises(InvalidDecision):
                check(answer(**{field: value}))

    def test_a_short_cannot_target_more_than_95_percent(self):
        self.assertEqual(check(answer(action="short", take_profit_pct=90)).take_profit_pct, 90)
        with self.assertRaises(InvalidDecision):
            check(answer(action="short", take_profit_pct=120))
        self.assertEqual(check(answer(take_profit_pct=120)).take_profit_pct, 120)  # fine for a buy


class Levels(unittest.TestCase):
    def test_buy_levels_bracket_the_price(self):
        d = Decision("buy", "why", "NVDA", stop_loss_pct=6, take_profit_pct=12)
        self.assertEqual(d.levels(100.0), {"stop_loss": 94.0, "price_target": 112.0})

    def test_short_levels_are_mirrored(self):
        d = Decision("short", "why", "NVDA", stop_loss_pct=6, take_profit_pct=12)
        self.assertEqual(d.levels(100.0), {"stop_loss": 106.0, "price_target": 88.0})

    def test_no_levels_without_percentages(self):
        self.assertEqual(Decision("buy", "why", "NVDA").levels(100.0), {})


class PositionSize(unittest.TestCase):
    def test_wanted_size_fits(self):
        self.assertEqual(size_position(5, {"open_signals": 3, "allocated_pct": 15}), (5, None))

    def test_capped_by_what_is_left(self):
        self.assertEqual(size_position(5, {"open_signals": 19, "allocated_pct": 97.5}), (2.5, None))

    def test_fully_allocated(self):
        size, why = size_position(5, {"open_signals": 20, "allocated_pct": 99.95})
        self.assertEqual(size, 0)
        self.assertIn("fully allocated", why)

    def test_open_signal_limit(self):
        size, why = size_position(0.5, {"open_signals": 100, "allocated_pct": 50})
        self.assertEqual(size, 0)
        self.assertIn("100 open signals", why)


if __name__ == "__main__":
    unittest.main()
