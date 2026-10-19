"""The model's answer as a validated decision, checked against Alpha Bazaar's signal rules before anything is sent.

The rule: a field that's missing gets a safe default; a field that's present but wrong rejects the whole decision
(no trade today, and the run log says why). The limits below are the ones POST /api/publisher/submit-pick enforces.
"""
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple

ACTIONS = ("buy", "short", "none")
TIME_HORIZONS = ("day", "swing", "1W", "2W", "1M", "3M", "6M", "1Y")
# The signal closes itself after this many calendar days (AB's auto_close_days)
HOLD_DAYS = {"day": 1, "swing": 10, "1W": 7, "2W": 14, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}
DEFAULT_HORIZON = "1W"
DEFAULT_CONFIDENCE = 5
REASONING_MAX = 2000  # AB's limit for a signal's reasoning
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")  # AB's symbol rule: AAPL, BRK.B, BF-B
MAX_OPEN_SIGNALS = 100  # AB: at most 100 open signals...
MAX_ALLOCATION = 100.0  # ...whose position sizes add up to at most 100%
MIN_POSITION = 0.1
STOP_RANGE = (0.5, 50.0)  # stop loss, % away from the entry price
TARGET_RANGE = (0.5, 200.0)  # take profit, % away from the entry price (a short can make at most 95%)
SHORT_TARGET_MAX = 95.0

_SYNONYMS = {"long": "buy", "hold": "none", "pass": "none", "skip": "none", "wait": "none",
             "no_trade": "none", "no trade": "none", "nothing": "none"}
_HORIZONS = {h.lower(): h for h in TIME_HORIZONS}
_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")

# JSON Schema for providers that can enforce one (every field present; null means "no value").
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "ticker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "time_horizon": {"anyOf": [{"type": "string", "enum": list(TIME_HORIZONS)}, {"type": "null"}]},
        "confidence": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "stop_loss_pct": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "take_profit_pct": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "ticker", "time_horizon", "confidence", "stop_loss_pct", "take_profit_pct", "reasoning"],
    "additionalProperties": False,
}


class InvalidDecision(ValueError):
    """The model's answer can't be used. The message says what was wrong."""


@dataclass
class Decision:
    action: str  # buy, short or none
    reasoning: str
    ticker: Optional[str] = None
    time_horizon: str = DEFAULT_HORIZON
    confidence: int = DEFAULT_CONFIDENCE
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    @property
    def is_trade(self) -> bool:
        return self.action in ("buy", "short")

    @property
    def hold_days(self) -> int:
        return HOLD_DAYS[self.time_horizon]

    def levels(self, price: float) -> Dict[str, float]:
        """stop_loss and price_target in dollars from a reference price. AB checks them against its live price:
        for a buy the target must be above it and the stop below it (the reverse for a short)."""
        out: Dict[str, float] = {}
        sign = 1 if self.action == "buy" else -1
        if self.stop_loss_pct:
            out["stop_loss"] = round(price * (1 - sign * self.stop_loss_pct / 100), 2)
        if self.take_profit_pct:
            out["price_target"] = round(price * (1 + sign * self.take_profit_pct / 100), 2)
        return out

    def summary(self) -> str:
        if not self.is_trade:
            return "NO TRADE"
        parts = [f"{self.action.upper()} {self.ticker}", self.time_horizon, f"confidence {self.confidence}/10"]
        if self.stop_loss_pct:
            parts.append(f"stop {self.stop_loss_pct:g}%")
        if self.take_profit_pct:
            parts.append(f"target {self.take_profit_pct:g}%")
        return " · ".join(parts)


def parse_json(reply: Any) -> Dict[str, Any]:
    """The first JSON object in a model's reply. Code fences and text around it are fine."""
    if isinstance(reply, Mapping):
        return dict(reply)
    if not isinstance(reply, str) or not reply.strip():
        raise InvalidDecision("the model's reply was empty")
    decoder = json.JSONDecoder()
    start = reply.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(reply, start)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = reply.find("{", start + 1)
    raise InvalidDecision("the model's reply had no JSON object in it")


def clean_text(text: str, limit: int) -> str:
    """Plain, single-spaced text of at most `limit` characters (cut at a word, with an ellipsis)."""
    text = _SPACE.sub(" ", _TAGS.sub("", text)).strip()
    text = "".join(ch for ch in text if ch.isprintable())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "…"


def _number(value: Any, field: str) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise InvalidDecision(f'"{field}" must be a number, not {value!r}')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidDecision(f'"{field}" must be a number, not {value!r}') from None
    if not math.isfinite(number):
        raise InvalidDecision(f'"{field}" must be a finite number')
    return number


def _horizon(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_HORIZON
    if isinstance(value, str) and value.strip().lower() in _HORIZONS:
        return _HORIZONS[value.strip().lower()]
    raise InvalidDecision(f'"time_horizon" must be one of {", ".join(TIME_HORIZONS)}, not {value!r}')


def _confidence(value: Any) -> int:
    number = _number(value, "confidence")
    if number is None:
        return DEFAULT_CONFIDENCE
    if number != int(number) or not 1 <= number <= 10:
        raise InvalidDecision(f'"confidence" must be a whole number from 1 to 10, not {value!r}')
    return int(number)


def _percent(value: Any, field: str, low: float, high: float) -> Optional[float]:
    number = _number(value, field)
    if not number:  # null or 0: no level
        return None
    if not low <= number <= high:
        raise InvalidDecision(f'"{field}" must be from {low:g} to {high:g} (percent from the entry price), '
                              f'not {value!r}')
    return round(number, 2)


def validate(reply: Any, *, candidates: Mapping[str, Any], held: Set[str],
             allow_any_ticker: bool = False) -> Decision:
    """A Decision from the model's reply, or InvalidDecision saying what's wrong with it.

    candidates: tickers the agent may trade today (they were in the context, with a price).
    held: tickers with an open signal already (one position per ticker).
    """
    data = parse_json(reply)

    action = data.get("action")
    if not isinstance(action, str) or not action.strip():
        raise InvalidDecision('"action" is missing: use buy, short or none')
    action = _SYNONYMS.get(action.strip().lower(), action.strip().lower())
    if action not in ACTIONS:
        raise InvalidDecision(f'"action" must be buy, short or none, not {data.get("action")!r}')

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise InvalidDecision('"reasoning" is missing: every decision is published with its explanation')
    reasoning = clean_text(reasoning, REASONING_MAX)
    if not reasoning:
        raise InvalidDecision('"reasoning" has no readable text')

    if action == "none":
        return Decision("none", reasoning)

    ticker = data.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        raise InvalidDecision(f'"ticker" is required to {action}')
    ticker = ticker.strip().lstrip("$").upper()
    if not TICKER_RE.match(ticker):
        raise InvalidDecision(f"{ticker!r} isn't a ticker symbol (1-10 letters, digits, '.' or '-')")
    if ticker in held:
        raise InvalidDecision(f"there's already an open signal on {ticker} (one position per ticker)")
    if not allow_any_ticker and ticker not in candidates:
        raise InvalidDecision(f"{ticker} isn't one of today's candidates (tickers in the data, with a price)")

    target_max = TARGET_RANGE[1] if action == "buy" else SHORT_TARGET_MAX
    return Decision(
        action=action,
        reasoning=reasoning,
        ticker=ticker,
        time_horizon=_horizon(data.get("time_horizon")),
        confidence=_confidence(data.get("confidence")),
        stop_loss_pct=_percent(data.get("stop_loss_pct"), "stop_loss_pct", *STOP_RANGE),
        take_profit_pct=_percent(data.get("take_profit_pct"), "take_profit_pct", TARGET_RANGE[0], target_max),
    )


def size_position(wanted: float, portfolio: Mapping[str, Any]) -> Tuple[float, Optional[str]]:
    """The position size AB will accept now, or (0, why not). AB caps open signals and total allocation."""
    open_signals = int(portfolio.get("open_signals") or 0)
    if open_signals >= MAX_OPEN_SIGNALS:
        return 0.0, f"Alpha Bazaar allows {MAX_OPEN_SIGNALS} open signals and this agent has {open_signals}"
    allocated = float(portfolio.get("allocated_pct") or 0.0)
    available = max(MAX_ALLOCATION - allocated, 0.0)
    size = round(min(wanted, available), 2)
    if size < MIN_POSITION:
        return 0.0, f"the paper portfolio is fully allocated ({allocated:.1f}% is in open signals)"
    return size, None
