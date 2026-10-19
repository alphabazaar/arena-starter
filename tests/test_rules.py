"""The no-key momentum rules, on fixture data shaped like Alpha Bazaar's."""
import unittest

from agent import context, rules
from agent.decision import validate
from tests.fakes import fixture


def build_context(positions=None):
    return {
        "market": context.trim_pulse(fixture("pulse.json")),
        "digest": context.trim_digest(fixture("digest.json")),
        "confluences": context.trim_confluences(fixture("smart_feed.json")),
        "sectors": context.trim_sectors(fixture("sectors.json")),
        "portfolio": {"open_signals": len(positions or []), "allocated_pct": 0.0, "available_pct": 100.0,
                      "positions": [{"ticker": t} for t in positions or []]},
    }


class MomentumRules(unittest.TestCase):
    def decide(self, ctx):
        candidates = context.candidates(ctx, min_price=5.0)
        held = {p["ticker"] for p in ctx["portfolio"]["positions"]}
        tradable = {t: v for t, v in candidates.items() if t not in held}
        reply = rules.decide(ctx, tradable)
        return reply, validate(reply, candidates=tradable, held=held)

    def test_buys_the_actionable_trend_first(self):
        # NIC is "caution" (+9%); WLTH runs further (+33%) but isn't actionable; ANDG is overextended (+179%)
        reply, decision = self.decide(build_context())
        self.assertEqual((decision.action, decision.ticker), ("buy", "NIC"))
        self.assertEqual(decision.confidence, 6)
        self.assertEqual((decision.stop_loss_pct, decision.take_profit_pct, decision.time_horizon), (8, 15, "1W"))
        self.assertIn("NIC is up 9.0%", decision.reasoning)

    def test_then_the_strongest_run_up(self):
        reply, decision = self.decide(build_context(positions=["NIC"]))
        self.assertEqual(decision.ticker, "WLTH")
        self.assertIn("up 33.2%", decision.reasoning)

    def test_skips_penny_stocks_shorts_overextended_and_losers(self):
        ctx = build_context(positions=["NIC", "WLTH"])
        _, decision = self.decide(ctx)
        # BEAT is under $5, TDY is a short, ANDG ran 179%, KGC is down: the index fallback buys the strongest ETF
        self.assertEqual(decision.ticker, "QQQ")
        self.assertEqual(decision.confidence, 4)

    def test_no_trade_when_nothing_qualifies(self):
        ctx = build_context(positions=["NIC", "WLTH", "QQQ", "SPY"])
        reply, decision = self.decide(ctx)  # DIA is down today
        self.assertEqual(decision.action, "none")
        self.assertIn("no trade today", decision.reasoning)

    def test_is_deterministic(self):
        ctx = build_context()
        self.assertEqual(self.decide(ctx)[0], self.decide(ctx)[0])


class Candidates(unittest.TestCase):
    def test_prices_minimum_and_sources(self):
        found = context.candidates(build_context(), min_price=5.0)
        self.assertNotIn("BEAT", found)  # $0.41
        self.assertEqual(found["NIC"]["price"], 162.6)  # the sector board's live price wins over the confluence's
        self.assertIn("Big Money move", found["MSFT"]["seen_in"])
        self.assertEqual(found["SPY"]["seen_in"], ["index ETF"])

    def test_short_side_is_normalized(self):
        sides = {c["ticker"]: c["side"] for c in context.trim_confluences(fixture("smart_feed.json"))}
        self.assertEqual((sides["TDY"], sides["NIC"]), ("short", "buy"))


if __name__ == "__main__":
    unittest.main()
