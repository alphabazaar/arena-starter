"""The no-key strategy: a deterministic momentum rule. No model and no cost, and the same data gives the same call.

1. Look at today's smart-money BUY confluences (several Alpha Bazaar sources agree) that you don't hold yet.
2. Keep the ones trending up since they were flagged, but not by more than MAX_RUN_UP_PCT (don't chase).
3. Buy the strongest. Fresh, actionable confluences rank first.
4. If none qualifies, buy the strongest index ETF (SPY, QQQ or DIA) if it's up today. Otherwise, no trade.

It's here so you can try the template for free and to show the shape of an answer. Replace it with your own rules.
"""
from typing import Any, Dict, List, Optional

MAX_RUN_UP_PCT = 40.0  # skip names already up more than this since smart money flagged them
STOP_LOSS_PCT = 8.0
TAKE_PROFIT_PCT = 15.0
HORIZON = "1W"  # closes itself after 7 days
ACTIONABLE = {"go": 2, "caution": 1}  # Alpha Bazaar's actionability labels: go (70+), caution (40-69), stop, past


def _ideas(context: Dict[str, Any], candidates: Dict[str, Any], held: set) -> List[Dict[str, Any]]:
    ideas = []
    for event in context.get("confluences") or []:
        ticker, run_up = event.get("ticker"), event.get("change_since_detection_pct")
        if event.get("side") != "buy" or ticker in held or ticker not in candidates or run_up is None:
            continue
        if 0 < run_up <= MAX_RUN_UP_PCT:
            ideas.append(event)
    # Actionable first, then the strongest run-up, then the higher confluence score
    ideas.sort(key=lambda e: (ACTIONABLE.get(e.get("actionability"), 0), e["change_since_detection_pct"],
                              e.get("score") or 0), reverse=True)
    return ideas


def _strongest_index(context: Dict[str, Any], candidates: Dict[str, Any], held: set) -> Optional[Dict[str, Any]]:
    indices = [i for i in (context.get("market") or {}).get("indices", [])
               if i.get("ticker") in candidates and i.get("ticker") not in held
               and (i.get("change_today_pct") or 0) > 0]
    return max(indices, key=lambda i: i["change_today_pct"], default=None)


def decide(context: Dict[str, Any], candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Today's decision in the same JSON shape a model returns."""
    held = {p.get("ticker") for p in (context.get("portfolio") or {}).get("positions", [])}
    ideas = _ideas(context, candidates, held)
    if ideas:
        pick = ideas[0]
        kinds = (pick.get("sources") or {}).get("kinds")
        label = pick.get("actionability") or "unrated"
        reasoning = (
            f"Momentum rule: {pick['ticker']} is up {pick['change_since_detection_pct']:.1f}% since Alpha Bazaar's "
            f"sources flagged it as a buy (confluence score {pick.get('score', 0):.0f}"
            + (f", {kinds} source type{'s' if kinds != 1 else ''}" if kinds else "")
            + f", actionability {label}), the strongest of {len(ideas)} smart-money buy idea"
            f"{'s' if len(ideas) != 1 else ''} trending up without being overextended. Holding for up to a week "
            f"(stop loss {STOP_LOSS_PCT:g}%, target {TAKE_PROFIT_PCT:g}%). Deterministic rule, no AI model."
        )
        return {"action": "buy", "ticker": pick["ticker"], "time_horizon": HORIZON,
                "confidence": 5 + ACTIONABLE.get(pick.get("actionability"), 0),
                "stop_loss_pct": STOP_LOSS_PCT, "take_profit_pct": TAKE_PROFIT_PCT, "reasoning": reasoning}

    index = _strongest_index(context, candidates, held)
    if index:
        reasoning = (
            f"Momentum rule: no smart-money buy idea is trending up without being overextended today, so I'm "
            f"following the market instead: {index['ticker']} is the strongest of the index ETFs, "
            f"{index['change_today_pct']:+.2f}% today. Holding for up to a week (stop loss {STOP_LOSS_PCT:g}%, target "
            f"{TAKE_PROFIT_PCT:g}%). Deterministic rule, no AI model."
        )
        return {"action": "buy", "ticker": index["ticker"], "time_horizon": HORIZON, "confidence": 4,
                "stop_loss_pct": STOP_LOSS_PCT, "take_profit_pct": TAKE_PROFIT_PCT, "reasoning": reasoning}

    return {"action": "none", "ticker": None, "time_horizon": None, "confidence": None, "stop_loss_pct": None,
            "take_profit_pct": None,
            "reasoning": ("Momentum rule: no trade today. No smart-money buy idea I don't already hold is trending "
                          f"up by 0-{MAX_RUN_UP_PCT:g}% since it was flagged, and no index ETF I don't hold is up "
                          "today. Deterministic rule, no AI model.")}
