"""What both adapters share: a framework's decisions in, Alpha Bazaar paper signals and one run-log entry out.

An adapter reads its framework's output and says, ticker by ticker, what the agent should hold on Alpha Bazaar
afterwards: an Intent. publish() compares that with the agent's open signals and does the least that gets there. It
closes what the framework left, opens what it asked for (checked against Alpha Bazaar's limits first), and explains
every decision, trade or not, in one public run-log entry. A decision Alpha Bazaar rejects becomes a line in that
entry, not a crash.

Paper trading only. Nothing the adapters or Alpha Bazaar produce is investment advice.
"""
import argparse
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from agent import config
from agent.ab import AB, ABError, redact
from agent.config import VERSION, ConfigError
from agent.decision import HOLD_DAYS, MIN_POSITION, REASONING_MAX, TICKER_RE, TIME_HORIZONS, size_position
from agent.main import META_MAX_BYTES, RUN_LOG_MAX, Report

try:
    from zoneinfo import ZoneInfo
    NEW_YORK = ZoneInfo("America/New_York")
except Exception:  # no time zone database (some Windows installs): use the computer's date
    NEW_YORK = None

ME = "/api/agents/me"
PICKS = "/api/publisher/my-picks"
SUBMIT = "/api/publisher/submit-pick"
CLOSE = "/api/performance/close-signal/{id}"
LOG = "/api/agents/me/log"

# What an Intent asks the agent to hold on the ticker afterwards
LONG = "long"              # a long: open one, after closing an open short
SHORT = "short"            # a short: open one, after closing an open long
FLAT = "flat"              # nothing: close whatever is open
EXIT_LONG = "exit_long"    # close an open long, if there is one (a short stays)
EXIT_SHORT = "exit_short"  # close an open short, if there is one (a long stays)
KEEP = "keep"              # no change
TARGETS = (LONG, SHORT, FLAT, EXIT_LONG, EXIT_SHORT, KEEP)

DEFAULT_SIZE = 5.0
DEFAULT_MAX_AGE_DAYS = 5
SIGNAL_SUMMARY_MAX = 1200  # of a signal's 2000 characters of reasoning; the headline before it takes the rest
LOG_SUMMARY_MAX = 600  # per ticker in the run-log entry; less when one entry covers many tickers
LEVELS = ("stop_loss", "price_target")
SIDE = {"buy": "long", "short": "short"}

_URL = re.compile(r"https?://([^/\s)\]>\"']+)[^\s)\]>\"']*")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.M)
_MARKDOWN = re.compile(r"\*\*|__|`+|^\s{0,3}#{1,6}\s+|^\s*[-*+•]\s+|^\s*>\s?", re.M)
_TAG = re.compile(r"</?[A-Za-z!][^>]*>")


class AdapterError(Exception):
    """The framework's output can't be used, or the framework can't run. The message says why and what to do."""


class RunFailed(AdapterError):
    """The framework started but its run failed (a data or model error), so there's nothing to publish."""


@dataclass
class Intent:
    """One framework decision on one ticker, and what it means the agent should hold on Alpha Bazaar afterwards."""

    ticker: str
    target: str  # one of TARGETS
    said: str  # the call in the framework's own terms, for the run log: "rated Buy", "decided sell 40 shares"
    headline: str  # the first sentence of a signal's reasoning: who decided what, and when
    summary: str = ""  # the framework's reasoning; cleaned, shortened and redacted before anything is sent
    why: str = ""  # when this leads to no trade: why (the run log says "no trade: <why>")
    confidence: Optional[int] = None  # 1-10, when the framework gives a conviction (None: Alpha Bazaar's default)
    size_hint: Optional[float] = None  # % of the portfolio the framework's own sizing implies (--size-from-fund)
    levels: Dict[str, float] = field(default_factory=dict)  # stop_loss and price_target in dollars, if it set them
    as_of: Optional[str] = None  # the date the framework decided for, YYYY-MM-DD
    call: str = ""  # the call in a word or two, for the run log's meta: "Buy", "sell 40", "+4.0%"
    minor: bool = False  # a non-decision such as a zero weight on a name nobody holds: listed briefly when idle


@dataclass
class Position:
    """An open signal on Alpha Bazaar."""

    id: str
    ticker: str
    side: str  # buy (a long) or short
    size: float
    opened: Optional[str] = None  # YYYY-MM-DD


@dataclass
class Settings:
    framework: str  # the profile's framework label: TradingAgents or ai-hedge-fund
    adapter: str  # names this adapter in the log files and the User-Agent
    source: str  # who decided, as the run log names it: "TradingAgents 0.7.1 (gpt-6-sol)"
    model: Optional[str] = None  # model label for the profile; None leaves the profile's as it is
    framework_version: Optional[str] = None
    horizon: str = "1W"
    size: float = DEFAULT_SIZE
    size_from_framework: bool = False
    use_levels: bool = False
    max_age_days: int = DEFAULT_MAX_AGE_DAYS
    sync_profile: bool = True


@dataclass
class Outcome:
    """What publish() planned and did for one intent."""

    intent: Intent
    closes: List[Position] = field(default_factory=list)  # open signals to close
    opens: Optional[str] = None  # buy or short: the new signal to publish once the closes are done
    closed: List[Position] = field(default_factory=list)
    done: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)  # what Alpha Bazaar refused, and what that stopped
    idle: str = ""  # why nothing was done
    body: Optional[Dict[str, Any]] = None  # the submit-pick body
    signal_id: Optional[str] = None

    @property
    def quiet(self) -> bool:
        return self.intent.minor and not (self.done or self.problems or self.closes or self.opens)

    def text(self) -> str:
        parts = self.done + self.problems
        if parts:
            return "; ".join(parts + ([self.idle] if self.idle else []))
        return f"no trade: {self.idle or 'nothing to do'}"


# --- small helpers --------------------------------------------------------------------------------------------------

def today_in_new_york() -> date:
    return datetime.now(NEW_YORK).date() if NEW_YORK else date.today()


def normalize_ticker(value: Any) -> str:
    return str(value or "").strip().lstrip("$").upper()


def clip(text: str, limit: int) -> str:
    """Plain, single-spaced text of at most `limit` characters, cut at a word with an ellipsis."""
    text = " ".join(_TAG.sub("", text).split())
    text = "".join(ch for ch in text if ch.isprintable())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "…"


def summarize(text: Any, limit: int) -> str:
    """Reasoning fit to publish: no markdown, links shortened to their site, keys masked, at most `limit` long."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, default=str) if isinstance(text, (dict, list)) else str(text)
    text = _MARKDOWN.sub("", _TABLE_RULE.sub(" ", text)).replace("|", " ")
    text = _URL.sub(lambda match: match.group(1), text)
    return clip(redact(text), limit)


def number(value: Any) -> Optional[float]:
    """A finite number from a number or numeric text ("1,234.50", "$12"), else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "").lstrip("$")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def confidence_from(value: Any, scale: float) -> Optional[int]:
    """A conviction on a 0-`scale` scale as Alpha Bazaar's 1-10 (rounded half up), or None."""
    amount = number(value)
    if amount is None:
        return None
    return int(min(max(math.floor(abs(amount) / scale * 10 + 0.5), 1), 10))


def horizon(value: Optional[str], default: str) -> str:
    if not value:
        return default
    for name in TIME_HORIZONS:
        if value.strip().lower() == name.lower():
            return name
    raise ConfigError(f"--horizon must be one of {', '.join(TIME_HORIZONS)}, not {value!r}")


def open_positions(answer: Dict[str, Any]) -> List[Position]:
    out = []
    for pick in answer.get("picks") or []:
        if pick.get("status") != "open":
            continue
        side = "short" if str(pick.get("action") or "").lower() in ("short", "sell") else "buy"
        out.append(Position(id=str(pick.get("id")), ticker=normalize_ticker(pick.get("ticker")), side=side,
                            size=number(pick.get("position_size")) or 0.0,
                            opened=str(pick.get("created_at") or "")[:10] or None))
    return out


def _json_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def fit_meta(meta: Dict[str, Any], calls: Dict[str, str]) -> Dict[str, Any]:
    """The run-log meta object within Alpha Bazaar's 2 KB, with as many of the per-ticker calls as fit."""
    meta = {key: value for key, value in meta.items() if value not in (None, "")}
    fitted: Dict[str, str] = {}
    for ticker, call in calls.items():
        if _json_size({**meta, "calls": {**fitted, ticker: call}, "calls_truncated": True}) > META_MAX_BYTES:
            meta["calls_truncated"] = True
            break
        fitted[ticker] = call
    return {**meta, "calls": fitted} if fitted else meta


# --- command line ---------------------------------------------------------------------------------------------------

def add_publish_args(parser: argparse.ArgumentParser, default_horizon: str) -> None:
    group = parser.add_argument_group("publishing")
    group.add_argument("--dry-run", action="store_true",
                       help="print exactly what would be sent to Alpha Bazaar, and send nothing")
    group.add_argument("--size", type=float,
                       help=f"%% of the paper portfolio per new signal, 0.1-100 (default: AB_POSITION_SIZE, else "
                            f"{DEFAULT_SIZE:g})")
    group.add_argument("--horizon", help=f"how long a new signal stays open at most: {', '.join(TIME_HORIZONS)} "
                                         f"(default: {default_horizon})")
    group.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                       help=f"skip decisions made more than this many days ago (default: {DEFAULT_MAX_AGE_DAYS}); "
                            "a signal enters at the price when it's sent")
    group.add_argument("--model", help="model label for your public profile (default: the model the output names)")
    group.add_argument("--no-profile", action="store_true",
                       help="leave your profile's framework and model labels as they are")
    group.add_argument("--log-dir", help="folder for this run's log files (default: logs)")


def position_size(args: argparse.Namespace, cfg: config.Config) -> float:
    if args.size is None:
        return cfg.position_size  # AB_POSITION_SIZE, checked by config.load(), else 5
    if not 0.1 <= args.size <= 100:
        raise ConfigError(f"--size must be from 0.1 to 100 (% of the paper portfolio), not {args.size:g}")
    return args.size


# --- planning -------------------------------------------------------------------------------------------------------

def _dedupe(intents: Sequence[Intent]) -> List[Intent]:
    """One intent per ticker: the latest decision wins (by date, then by order)."""
    latest: Dict[str, Intent] = {}
    for intent in intents:
        previous = latest.get(intent.ticker)
        if previous is None or (intent.as_of or "") >= (previous.as_of or ""):
            latest[intent.ticker] = intent
    return [intent for intent in intents if latest.get(intent.ticker) is intent]


def _age_problem(intent: Intent, settings: Settings, today: date) -> Optional[str]:
    if not intent.as_of:
        return None
    try:
        decided = date.fromisoformat(intent.as_of[:10])
    except ValueError:
        return f"its date {intent.as_of!r} isn't a date (YYYY-MM-DD)"
    days = (today - decided).days
    if days > settings.max_age_days:
        return (f"it was decided for {decided.isoformat()}, {days} days ago, too long ago to publish as a new call "
                "(a signal enters at the price when it's sent; --max-age-days changes the limit)")
    return None


def plan(intent: Intent, held: Sequence[Position], settings: Settings, today: date) -> Outcome:
    """What to do for one intent, given the ticker's open signals."""
    outcome = Outcome(intent)
    if not TICKER_RE.match(intent.ticker):
        outcome.idle = f"{intent.ticker!r} isn't a ticker symbol Alpha Bazaar takes (1-10 letters, digits, . or -)"
        return outcome
    if intent.target != KEEP:
        problem = _age_problem(intent, settings, today)
        if problem:
            outcome.idle = problem
            return outcome
    longs = [p for p in held if p.side == "buy"]
    shorts = [p for p in held if p.side == "short"]
    target = intent.target
    if target == LONG:
        outcome.closes, outcome.opens = shorts, None if longs else "buy"
    elif target == SHORT:
        outcome.closes, outcome.opens = longs, None if shorts else "short"
    elif target == FLAT:
        outcome.closes = longs + shorts
    elif target == EXIT_LONG:
        outcome.closes = longs
    elif target == EXIT_SHORT:
        outcome.closes = shorts
    if not outcome.closes and not outcome.opens:
        stays = "; ".join(f"the open {SIDE[p.side]} from {p.opened or 'earlier'} stays open" for p in longs + shorts)
        why = "" if target in (LONG, SHORT) else intent.why
        outcome.idle = "; ".join(part for part in (why, stays) if part) or "nothing to do"
    return outcome


def signal_body(intent: Intent, action: str, size: float, settings: Settings) -> Dict[str, Any]:
    """The POST /api/publisher/submit-pick body. Everything in it is public."""
    summary = summarize(intent.summary, SIGNAL_SUMMARY_MAX)
    reasoning = clip(redact(f"{intent.headline} {summary}".strip()), REASONING_MAX)
    body: Dict[str, Any] = {"ticker": intent.ticker, "action": action, "reasoning": reasoning,
                            "time_horizon": settings.horizon, "position_size": size,
                            "auto_close_days": HOLD_DAYS[settings.horizon]}
    if intent.confidence is not None:
        body["confidence"] = intent.confidence
    if settings.use_levels:
        body.update(_levels(intent.levels, action))
    return body


def _levels(levels: Dict[str, float], action: str) -> Dict[str, float]:
    """The framework's stop and target in dollars, when they're positive and on the right sides of each other.
    Alpha Bazaar checks them against its live price; a signal with a level on the wrong side goes again without."""
    out = {key: round(value, 2) for key, value in levels.items() if key in LEVELS and value and value > 0}
    stop, target = out.get("stop_loss"), out.get("price_target")
    if stop and target and ((action == "buy" and stop >= target) or (action == "short" and stop <= target)):
        return {}
    return out


# --- publishing -----------------------------------------------------------------------------------------------------

def _show(report: Report, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> None:
    shown = "" if body is None else "\n" + json.dumps(body, indent=2, ensure_ascii=False)
    report.say(f"  Would send: {method} {path}{shown}")


def _submit(ab: AB, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Publish a signal. A stop or target on the wrong side of Alpha Bazaar's live price is a 400 that saved nothing,
    so that one is sent again without levels."""
    try:
        return ab.post(SUBMIT, body), body
    except ABError as e:
        if e.status != 400 or not any(key in body for key in LEVELS) or not any(key in e.message for key in LEVELS):
            raise
        bare = {key: value for key, value in body.items() if key not in LEVELS}
        return ab.post(SUBMIT, bare), bare


def run_log_text(outcomes: Sequence[Outcome], settings: Settings, notes: Sequence[str]) -> str:
    """One entry for the run: each ticker's call, what was done about it, and why. At most 4000 characters."""
    dates = sorted({o.intent.as_of[:10] for o in outcomes if o.intent.as_of})
    when = f", decisions for {dates[0]}" if len(dates) == 1 else ""
    header = f"{settings.source}{when}. Published by the arena-starter adapter (paper trading)."
    notes = [clip(redact(note), 400) for note in notes]
    shown = [o for o in outcomes if not o.quiet]
    quiet = [o.intent.ticker for o in outcomes if o.quiet]
    lines = [f"{o.intent.ticker}: {o.intent.said} → {o.text()}." for o in shown]
    tail = [f"No position and none wanted: {', '.join(quiet)}."] if quiet else []
    explained = [i for i, o in enumerate(shown) if summarize(o.intent.summary, LOG_SUMMARY_MAX)]
    if explained:  # the reasons share what's left of the 4000 characters
        room = RUN_LOG_MAX - len("\n".join([header, *notes, *lines, *tail])) - len(" Why: ") * len(explained)
        budget = min(LOG_SUMMARY_MAX, room // len(explained))
        if budget >= 60:
            for i in explained:
                lines[i] += f" Why: {summarize(shown[i].intent.summary, budget)}"
    text = redact("\n".join([header, *notes, *lines, *tail]))
    if len(text) > RUN_LOG_MAX:
        cut = text[: RUN_LOG_MAX - 2]
        text = (cut.rsplit("\n", 1)[0] if "\n" in cut else cut) + "\n…"
    return text


def publish(intents: Sequence[Intent], settings: Settings, cfg: config.Config, report: Report, *,
            notes: Sequence[str] = (), failures: int = 0, today: Optional[date] = None) -> int:
    """Bring the agent's open signals in line with the intents, then write one run-log entry. Returns the exit code:
    0 when everything was done or deliberately not done, 1 when Alpha Bazaar refused something (or `failures`)."""
    today = today or today_in_new_york()
    dry = cfg.dry_run
    rec = report.record
    rec.update({"adapter": settings.adapter, "framework": settings.framework,
                "framework_version": settings.framework_version, "model": settings.model, "api_base": cfg.api_base,
                "dry_run": dry})
    if not cfg.api_key and not dry:
        raise ConfigError(f"AB_API_KEY isn't set. Register once with `python register.py --name <agent> --framework "
                          f"{settings.framework}` and save the key, or try this first with --dry-run.")
    ab = AB(cfg.api_base, cfg.api_key)
    ab.session.headers["User-Agent"] += f" adapter/{settings.adapter}"
    me = (ab.get(ME, auth=True).get("agent") or {}) if cfg.api_key else {}
    name = me.get("username")
    rec["agent"] = name
    report.say(f"Alpha Bazaar adapter for {settings.framework}, arena-starter {VERSION} "
               "(paper trading only, not investment advice)")
    report.say(f"{name or 'Not registered yet'} on {urlsplit(cfg.api_base).netloc} · {settings.source} · "
               f"{'DRY RUN, nothing will be sent' if dry else 'live'}")

    held = open_positions(ab.get(PICKS, auth=True, status="open", limit=100)) if cfg.api_key else []
    portfolio = {"open_signals": len(held), "allocated_pct": round(sum(p.size for p in held), 2)}
    if cfg.api_key:
        report.say(f"Open signals: {portfolio['open_signals']} · {portfolio['allocated_pct']:g}% of the paper "
                   "portfolio allocated")
    else:
        report.say("No AB_API_KEY, so your open signals aren't known: this assumes there are none")
    for note in notes:
        report.say(note)
    by_ticker: Dict[str, List[Position]] = {}
    for position in held:
        by_ticker.setdefault(position.ticker, []).append(position)
    outcomes = [plan(intent, by_ticker.get(intent.ticker, []), settings, today) for intent in _dedupe(intents)]
    if not outcomes and not notes:  # nothing decided: write nothing, not even the profile's labels
        report.say("Nothing to publish: the output had no decisions in it.")
        return 1 if failures else 0

    # 1. The profile says what's deciding (the Arena compares agents by framework and model)
    wanted = {"framework": settings.framework, **({"model": settings.model} if settings.model else {})}
    change = {key: value for key, value in wanted.items() if (me.get(key) or "") != value}
    if settings.sync_profile and change:
        if dry:
            report.say("Profile labels" + (":" if cfg.api_key else " (sent only where your profile's differ):"))
            _show(report, "PATCH", ME, change)
        else:
            try:
                ab.patch(ME, change)
                report.say("Profile: " + ", ".join(f"{key} set to {value}" for key, value in change.items()))
            except ABError as e:
                report.warn(f"couldn't update the profile's labels: {e}")

    # 2. Close what the framework left. A close during market hours also frees room for new signals.
    for outcome in outcomes:
        for position in outcome.closes:
            what = f"the open {SIDE[position.side]} (opened {position.opened or 'earlier'})"
            report.say(f"{outcome.intent.ticker}: close {what}")
            path = CLOSE.format(id=position.id)
            if dry:
                _show(report, "POST", path)
            else:
                try:
                    answer = ab.post(path)
                except ABError as e:
                    outcome.problems.append(f"tried to close {what}, but Alpha Bazaar answered: "
                                            f"{e.message.rstrip('.')}")
                    report.warn(f"{outcome.intent.ticker}: couldn't close signal {position.id}: {e}")
                    continue
                immediate = "immediate" in str(answer.get("action") or "")
                price = number(answer.get("exit_price"))
                report.say(f"  Closed signal {position.id}" + (f" at ${price:.2f}" if immediate and price else
                                                                ": it exits at the next market open"))
                if immediate:  # a close queued for the next open counts against the limits until then
                    portfolio["open_signals"] -= 1
                    portfolio["allocated_pct"] = round(max(portfolio["allocated_pct"] - position.size, 0.0), 2)
            outcome.closed.append(position)
            outcome.done.append(f"closed {what}")

    # 3. Open what it asked for, strongest conviction first (room and the weekly limit can run out)
    blocked: Optional[str] = None
    for outcome in sorted((o for o in outcomes if o.opens), key=lambda o: -(o.intent.confidence or 0)):
        intent, side = outcome.intent, SIDE[outcome.opens]
        if len(outcome.closed) < len(outcome.closes):
            outcome.problems.append(f"so no {side} was published (one position per ticker)")
            continue
        if blocked:
            outcome.problems.append(f"wanted a {side}, but {blocked}")
            continue
        wanted_size = settings.size
        if settings.size_from_framework and intent.size_hint:
            wanted_size = round(min(max(intent.size_hint, MIN_POSITION), 100.0), 2)
        size, why_not = size_position(wanted_size, portfolio)
        if why_not:
            outcome.idle = f"wanted a {side}, but {why_not}"
            continue
        body = signal_body(intent, outcome.opens, size, settings)
        report.say(f"{intent.ticker}: publish a {side}, {size:g}% for up to {HOLD_DAYS[settings.horizon]} days")
        if dry:
            _show(report, "POST", SUBMIT, body)
        else:
            try:
                answer, body = _submit(ab, body)
            except ABError as e:
                outcome.problems.append(f"tried to publish a {side}, but Alpha Bazaar rejected it: "
                                        f"{e.message.rstrip('.')}")
                report.warn(f"{intent.ticker}: Alpha Bazaar didn't accept the signal: {e}")
                if e.status == 429:
                    blocked = f"Alpha Bazaar's limit stopped the signals before it ({e.message.rstrip('.')})"
                continue
            outcome.signal_id = answer.get("pick_id")
            report.say(f"  Published signal {outcome.signal_id}: entry price "
                       + ("captured" if answer.get("entry_price_status") == "captured" else "set at the next open"))
        outcome.body = body
        portfolio["open_signals"] += 1
        portfolio["allocated_pct"] = round(portfolio["allocated_pct"] + size, 2)
        confidence = f", confidence {intent.confidence}/10" if intent.confidence is not None else ""
        levels = "".join(f", {'stop' if key == 'stop_loss' else 'target'} ${body[key]:.2f}"
                         for key in LEVELS if key in body)
        outcome.done.append(f"published a {side}, {size:g}% of the paper portfolio for up to "
                            f"{HOLD_DAYS[settings.horizon]} days ({settings.horizon}){confidence}{levels}")

    rejected = sum(1 for o in outcomes if o.problems)
    exit_code = 1 if rejected or failures else 0
    rec["outcomes"] = [{"ticker": o.intent.ticker, "target": o.intent.target, "call": o.intent.call,
                        "result": o.text(), "signal_id": o.signal_id, "body": o.body} for o in outcomes]
    if outcomes:
        report.say("Decisions (as they'd be published if Alpha Bazaar accepts every step):" if dry else "Decisions:")
    for outcome in outcomes:
        if not outcome.quiet:
            report.say(f"  {outcome.intent.ticker}: {outcome.intent.said} → {outcome.text()}")
    quiet = [o.intent.ticker for o in outcomes if o.quiet]
    if quiet:
        report.say(f"  No position and none wanted: {', '.join(quiet)}")

    # 4. One run-log entry for the whole run, trade or not ("no trade because..." counts)
    single = outcomes[0].intent.ticker if len(outcomes) == 1 and TICKER_RE.match(outcomes[0].intent.ticker) else None
    meta = fit_meta({
        "adapter": f"arena-starter/{VERSION} {settings.adapter}", "framework": settings.framework,
        "framework_version": settings.framework_version, "model": settings.model,
        "as_of": max((o.intent.as_of[:10] for o in outcomes if o.intent.as_of), default=None),
        "published": sum(1 for o in outcomes if o.body), "closed": sum(len(o.closed) for o in outcomes),
        "rejected": rejected,
    }, {o.intent.ticker[:12]: o.intent.call[:24] for o in outcomes if o.intent.call})
    entry_body = {"kind": "decision", "content": run_log_text(outcomes, settings, notes), "ticker": single,
                  "meta": meta}
    if dry:
        report.say("Run log entry (as written when Alpha Bazaar accepts every step; a refusal is written in instead):")
        _show(report, "POST", LOG, entry_body)
        report.say("Dry run: nothing was sent to Alpha Bazaar.")
    else:
        try:
            entry = ab.post(LOG, entry_body)
            rec["run_log"] = {"seq": entry.get("seq"), "entry_hash": entry.get("entry_hash")}
            report.say(f"Run log: entry #{entry.get('seq')} written (hash {str(entry.get('entry_hash'))[:16]}...)")
        except ABError as e:
            exit_code = 1
            report.warn(f"couldn't write the run log: {e}")
        _attestations(ab, outcomes, cfg, report)

    report.summary += [f"### Alpha Bazaar: {settings.framework} decisions" + (" (dry run)" if dry else ""),
                       *[f"- **{o.intent.ticker}** {o.intent.said} → {o.text()}" for o in outcomes if not o.quiet]]
    if name and cfg.web_base:
        page = f"{cfg.web_base}/arena/agents/{name}"
        report.say(f"Watch it: {page} (profile, signals and run log)")
        report.say(f"README badge: [![Alpha Bazaar]({cfg.api_base}/api/agents/{name}/badge.svg)]({page})")
        report.summary.append(f"[Profile and run log]({page})")
    if name:
        report.say(f"Run log (API): {cfg.api_base}/api/agents/{name}/log (add /verify to check its hash chain)")
    return exit_code


def _attestations(ab: AB, outcomes: Sequence[Outcome], cfg: config.Config, report: Report) -> None:
    """Print each new signal's attestation code: anyone can look it up."""
    new = {o.signal_id for o in outcomes if o.signal_id}
    if not new:
        return
    try:
        for pick in ab.get(PICKS, auth=True, status="open", limit=100).get("picks", []):
            if pick.get("id") in new and pick.get("attestation_code"):
                report.say(f"  Attestation for {pick.get('ticker')}: {pick['attestation_code']} "
                           f"({cfg.api_base}/api/verify/{pick['attestation_code']})")
    except ABError:
        pass  # nice to have
