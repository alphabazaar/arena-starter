"""Market context from Alpha Bazaar, trimmed to what a model needs for one decision.

One call each per run, all served from Alpha Bazaar's caches: the market pulse (index ETFs), the daily digest
(brief, headlines, Big Money and insider moves, earnings), smart-money confluences and sector sentiment. With a
key it also reads your open signals and your Open Arena standing.
"""
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agent.ab import AB, ABError
from agent.decision import MAX_ALLOCATION, TICKER_RE

CONFLUENCE_LIMIT = 10  # the MCP list_confluences default, so the request shares Alpha Bazaar's cache


def _num(value: Any, digits: int = 2) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, digits)


def _date(value: Any) -> Optional[str]:
    return value[:10] if isinstance(value, str) and len(value) >= 10 else None


def _clip(text: Any, limit: int) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _side(value: Any) -> Optional[str]:
    side = str(value or "").strip().lower()
    return {"sell": "short", "long": "buy"}.get(side, side) or None


def _compact(value: Any) -> Any:
    """Drop None and empty values so the prompt stays short."""
    if isinstance(value, dict):
        out = {k: _compact(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [_compact(v) for v in value if v not in (None, "", [], {})]
    return value


def trim_pulse(data: Dict[str, Any]) -> Dict[str, Any]:
    return _compact({
        "status": data.get("market_status"),
        "as_of": data.get("as_of"),
        "indices": [{"ticker": i.get("symbol"), "name": i.get("label"), "price": _num(i.get("price")),
                     "change_today_pct": _num(i.get("change_pct"))} for i in data.get("indices") or []],
        "news": [_clip(n.get("title"), 160) for n in (data.get("market_news") or [])[:5]],
    })


def trim_digest(data: Dict[str, Any]) -> Dict[str, Any]:
    brief = data.get("daily_brief") or {}
    return _compact({
        "brief": _clip(brief.get("text"), 1500),
        "headlines": [{"title": _clip(h.get("title"), 160), "tickers": h.get("tickers")}
                      for h in (data.get("headlines") or [])[:6]],
        "big_money_moves": [{
            "ticker": m.get("ticker"), "action": m.get("action"),
            "who": m.get("display_name") or m.get("publisher_username"),
            "kind": m.get("source_kind") or m.get("publisher_type"), "value": m.get("value_range"),
            "date": _date(m.get("created_at")), "price": _num(m.get("current_price")),
            "change_since_pct": _num(m.get("change_pct")),
        } for m in (data.get("big_money_moves") or [])[:8]],
        "insider_moves": [{
            "ticker": i.get("ticker"), "insider": i.get("insider"), "title": i.get("title"),
            "action": i.get("action"), "value": i.get("value"), "date": _date(i.get("transaction_date")),
        } for i in (data.get("insider_moves") or [])[:6]],
        "earnings_today": [{
            "ticker": e.get("ticker"), "company": e.get("company_name"), "timing": e.get("timing"),
            "eps_estimate": e.get("eps_estimate"), "eps_actual": e.get("eps_actual"),
            "change_today_pct": _num(e.get("daily_change_pct")),
        } for e in (data.get("earnings_today") or [])[:8]],
    })


def trim_confluences(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Smart-money confluences: tickers where several independent sources point the same way."""
    out = []
    for event in (data.get("events") or [])[:CONFLUENCE_LIMIT]:
        mix = event.get("composition") or {}
        market = event.get("market") or {}
        out.append(_compact({
            "ticker": str(event.get("ticker") or "").upper(),
            "side": _side(event.get("side")),
            "score": _num(event.get("score"), 1),
            "actionability": (event.get("actionability") or {}).get("label"),
            "sources": {"total": mix.get("source_count"), "kinds": mix.get("unique_types_count"),
                        "politicians": mix.get("politician_count"), "funds_13f": mix.get("whale_count"),
                        "analysts": mix.get("analyst_count"), "traders": mix.get("publisher_count")},
            "price": _num(market.get("current_price")),
            "change_since_detection_pct": _num(event.get("price_change_pct")),
            "detected": _date(event.get("detected_at")),
            "days_active": event.get("days_active"),
            "sector": market.get("sector"),
            "narrative": _clip((event.get("trade_context") or {}).get("narrative"), 240),
        }))
    return out


def trim_sectors(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [_compact({
        "sector": s.get("sector"), "sentiment": s.get("sentiment"), "bullish_pct": s.get("bullish_pct"),
        "narrative": _clip(s.get("narrative"), 240),
        "tickers": [{"ticker": str(t.get("ticker") or "").upper(), "side": _side(t.get("side")),
                     "price": _num(t.get("price")), "change_today_pct": _num(t.get("price_change_pct"))}
                    for t in (s.get("tickers") or [])[:6]],
    }) for s in (data.get("sectors") or [])[:12]]


def trim_portfolio(data: Dict[str, Any]) -> Dict[str, Any]:
    """Your open signals and how much of the paper portfolio they use (AB allows 100% and 100 signals)."""
    picks = [p for p in data.get("picks") or [] if p.get("status") == "open"]
    allocated = round(sum(_num(p.get("position_size")) or 0.0 for p in picks), 2)
    return {
        "open_signals": len(picks) + (1 if data.get("has_more") else 0),
        "allocated_pct": allocated,
        "available_pct": round(max(MAX_ALLOCATION - allocated, 0.0), 2),
        "positions": [_compact({
            "ticker": str(p.get("ticker") or "").upper(), "action": p.get("action"),
            "size_pct": _num(p.get("position_size")), "entry_price": _num(p.get("entry_price")),
            "current_price": _num(p.get("current_price")), "unrealized_pct": _num(p.get("unrealized_pnl_percent")),
            "opened": _date(p.get("created_at")), "closes_after_days": p.get("auto_close_days"),
            "stop_loss": _num(p.get("stop_loss")), "price_target": _num(p.get("price_target")),
        }) for p in picks],
    }


def trim_arena(me: Dict[str, Any]) -> Dict[str, Any]:
    """Where you stand in the Open Arena (GET /api/arena/me)."""
    season = me.get("open_season") or None
    current = (me.get("seasons") or [None])[0] or {}
    standing = current.get("standing") or {}
    return _compact({
        "open_season": season and {k: season.get(k) for k in ("season", "name", "status", "starts_at", "ends_at",
                                                               "enrolled")},
        "standing": standing and {"season": current.get("season"), "rank": standing.get("rank"),
                                  "signals": standing.get("signals"), "signals_needed": standing.get("signals_needed"),
                                  "avg_return_pct": standing.get("avg_return"), "ranked": standing.get("ranked")},
        "next_step": (me.get("next_step") or {}).get("action"),
    })


def trim_seasons(data: Dict[str, Any]) -> Dict[str, Any]:
    """The next season's dates (GET /api/arena/seasons), for when no season is open for entries yet."""
    upcoming = data.get("upcoming") or {}
    return _compact({"upcoming": {k: upcoming.get(k) for k in ("season", "name", "starts_at", "ends_at")}})


def gather(ab: AB, *, authed: bool) -> Dict[str, Any]:
    """Everything the agent looks at today. A public section that fails is skipped (noted in "errors"); your own
    positions can't be skipped when trading, because the position limits depend on them."""
    ctx: Dict[str, Any] = {"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "errors": []}

    def read(name: str, path: str, trim: Callable[[Any], Any], **params: Any) -> Any:
        try:
            return trim(ab.get(path, **params))
        except ABError as e:
            ctx["errors"].append(f"{name}: {e}")
            return None

    ctx["market"] = read("market pulse", "/api/market/pulse", trim_pulse)
    ctx["digest"] = read("daily digest", "/api/feed/daily-digest", trim_digest)
    ctx["confluences"] = read("confluences", "/api/confluence/smart-feed", trim_confluences,
                              limit=CONFLUENCE_LIMIT, include_total="false")
    ctx["sectors"] = read("sector sentiment", "/api/sectors/intelligence", trim_sectors)
    if authed:
        ctx["portfolio"] = trim_portfolio(ab.get("/api/publisher/my-picks", auth=True, status="open", limit=100))
        ctx["arena"] = read("arena standing", "/api/arena/me", trim_arena, auth=True)
    else:
        ctx["portfolio"] = {"open_signals": 0, "allocated_pct": 0.0, "available_pct": MAX_ALLOCATION,
                            "positions": [], "note": "no AB_API_KEY, so your positions aren't known"}
        ctx["arena"] = None
    if not (ctx.get("arena") or {}).get("open_season"):
        seasons = read("arena seasons", "/api/arena/seasons", trim_seasons)
        if seasons:
            ctx["arena"] = {**(ctx.get("arena") or {}), **seasons}
    return ctx


def candidates(ctx: Dict[str, Any], min_price: float) -> Dict[str, Dict[str, Any]]:
    """Tickers the agent may trade today: every ticker in the context with a known price of at least min_price.
    Freshest prices first: index ETFs and the sector board (live, cached minutes), then confluences and Big Money."""
    found: Dict[str, Dict[str, Any]] = {}

    def add(ticker: Any, price: Any, where: str) -> None:
        symbol = str(ticker or "").strip().upper()
        value = _num(price)
        if not TICKER_RE.match(symbol) or value is None or value <= 0 or value < min_price:
            return
        entry = found.setdefault(symbol, {"price": value, "seen_in": []})
        if where not in entry["seen_in"]:
            entry["seen_in"].append(where)

    for index in (ctx.get("market") or {}).get("indices", []):
        add(index.get("ticker"), index.get("price"), "index ETF")
    for sector in ctx.get("sectors") or []:
        for row in sector.get("tickers", []):
            add(row.get("ticker"), row.get("price"), f"sector board ({sector.get('sector')})")
    for event in ctx.get("confluences") or []:
        add(event.get("ticker"), event.get("price"), f"{event.get('side')} confluence")
    for move in (ctx.get("digest") or {}).get("big_money_moves", []):
        add(move.get("ticker"), move.get("price"), "Big Money move")
    return found
