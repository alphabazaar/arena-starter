"""Publish TradingAgents decisions to Alpha Bazaar as paper signals, with a public run log.

  python -m adapters.tradingagents --ticker NVDA --rating Buy --reasoning "..."   # a decision you already have
  python -m adapters.tradingagents --file full_states_log_2026-11-12.json        # a saved TradingAgents run
  python -m adapters.tradingagents --latest NVDA,AAPL   # the newest saved run for each ticker
  python -m adapters.tradingagents --run NVDA,AAPL      # run TradingAgents now (installed separately), then publish

Built against TradingAgents v0.7.1 (TauricResearch/TradingAgents@c454491, 2026-10-27). There,
TradingAgentsGraph(config=...).propagate(ticker, date) returns (final_state, rating): the Portfolio Manager's rating on
a five-tier scale (Buy, Overweight, Hold, Underweight, Sell), or REVIEW when it can't be read. Every run, from the
CLI or from Python, also saves final_state to <results_dir>/<TICKER>/TradingAgentsStrategy_logs/
full_states_log_<date>.json, with final_rating, final_trade_decision (the decision in Markdown) and run_settings
(version and models). Older versions' BUY, HOLD and SELL read as Buy, Hold and Sell.

How a rating maps to Alpha Bazaar (see the README): Buy and Overweight open a long. Hold changes nothing. Underweight
and Sell close an open long, since TradingAgents means them as reduce or exit; Sell opens a short only with
--sell-as-short. Every decision, trade or not, is explained in the run log.
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adapters import common
from adapters.common import EXIT_LONG, KEEP, LONG, SHORT, AdapterError, Intent, RunFailed
from agent import config
from agent.ab import AB, ABError, redact
from agent.config import ConfigError
from agent.main import Report
from agent.providers import model_label

FRAMEWORK = "TradingAgents"
BUILT_FOR = "0.7.1"
DEFAULT_HORIZON = "1W"  # TradingAgents grades its own calls over 5 trading days (its holding_period_days)
RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
REVIEW = "REVIEW"
INSTALL = 'pip install "tradingagents @ git+https://github.com/TauricResearch/TradingAgents@v0.7.1"'
ANALYSTS = {"market": "market", "sentiment": "social", "social": "social", "news": "news",
            "fundamentals": "fundamentals"}
STATE_LOG = re.compile(r"full_states_log_(\d{4}-\d{2}-\d{2})\.json$")

_RATING_LINE = re.compile(
    r"^[\s>*_#\-•]*(?:(?:final|our|overall|my|recommended|updated|revised|new|current)\s+)*rating\s*\**\s*"
    r"[:：\-–—]\s*\**\s*(buy|overweight|hold|underweight|sell)\b", re.I | re.M)
_PROPOSAL = re.compile(r"final\s+transaction\s+proposal\s*[:：]\s*\**\s*(buy|hold|sell)\b", re.I)
_SECTION = r"\*\*{name}\*\*\s*[:：]\s*(.+?)(?=\n\s*\*\*[^*\n]{{2,40}}\*\*\s*[:：]|\Z)"


@dataclass
class Decision:
    """One TradingAgents run, reduced to what Alpha Bazaar needs."""

    ticker: str
    rating: str  # Buy, Overweight, Hold, Underweight, Sell or REVIEW
    as_of: Optional[str] = None  # the analysis date, YYYY-MM-DD
    summary: str = ""
    levels: Dict[str, float] = field(default_factory=dict)  # price_target (Portfolio Manager), stop_loss (Trader)
    model: Optional[str] = None  # the deep-thinking model: the Portfolio Manager's
    version: Optional[str] = None
    said: Optional[str] = None  # what the output said when it wasn't a rating


def normalize_rating(value: Any) -> Optional[str]:
    """Buy, Overweight, Hold, Underweight, Sell or REVIEW from any case (older versions say BUY, HOLD, SELL)."""
    text = str(value or "").strip().strip("*").strip().lower()
    return {r.lower(): r for r in (*RATINGS, REVIEW)}.get(text)


def rating_from_text(text: str) -> Optional[str]:
    """The rating a decision's text states: its "Rating: X" line, or an older version's "FINAL TRANSACTION PROPOSAL"."""
    match = _RATING_LINE.search(text or "") or _PROPOSAL.search(text or "")
    return match.group(1).capitalize() if match else None


def section(text: str, name: str) -> str:
    """The body of a "**Name**: ..." section of TradingAgents' Markdown, up to the next one."""
    match = re.search(_SECTION.format(name=re.escape(name)), text or "", re.S | re.I)
    return match.group(1).strip() if match else ""


def from_state(state: Dict[str, Any], settings: Optional[Dict[str, Any]] = None,
               rating: Optional[str] = None) -> Decision:
    """A Decision from a run's final state (what propagate() returns, and what the state log holds)."""
    ticker = common.normalize_ticker(state.get("company_of_interest") or state.get("ticker"))
    if not ticker:
        raise AdapterError("the TradingAgents state has no company_of_interest (ticker)")
    text = str(state.get("final_trade_decision") or "")
    found = normalize_rating(rating) or normalize_rating(state.get("final_rating")) or rating_from_text(text)
    settings = state.get("run_settings") or settings or {}
    summary = section(text, "Executive Summary") or section(text, "Investment Thesis") or _RATING_LINE.sub("", text)
    levels = {"price_target": common.number(section(text, "Price Target")),
              "stop_loss": common.number(section(str(state.get("trader_investment_plan") or ""), "Stop Loss"))}
    return Decision(ticker=ticker, rating=found or REVIEW, as_of=_date(state.get("trade_date")), summary=summary,
                    levels={key: value for key, value in levels.items() if value},
                    model=settings.get("deep_think_llm"), version=settings.get("version"))


def from_simple(item: Dict[str, Any]) -> Decision:
    """A Decision from a small JSON object: {"ticker", "rating" (or "decision"/"signal"), "reasoning", "date"}."""
    ticker = common.normalize_ticker(item.get("ticker") or item.get("symbol"))
    if not ticker:
        raise AdapterError(f"a decision has no ticker: {json.dumps(item)[:200]}")
    said = next((item[key] for key in ("rating", "decision", "signal", "action") if item.get(key)), None)
    rating = normalize_rating(said)
    levels = {key: common.number(item.get(key)) for key in common.LEVELS}
    return Decision(ticker=ticker, rating=rating or REVIEW, said=None if rating else str(said),
                    as_of=_date(item.get("date") or item.get("trade_date") or item.get("as_of")),
                    summary=str(item.get("reasoning") or item.get("summary") or item.get("executive_summary") or ""),
                    levels={key: value for key, value in levels.items() if value},
                    model=item.get("model"), version=item.get("version"))


def _date(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}", text) else None


def parse(data: Any, where: str) -> List[Decision]:
    """Decisions from TradingAgents output: a state log, a list of them, or small decision objects."""
    if isinstance(data, dict) and isinstance(data.get("decisions"), list):
        data = data["decisions"]
    if isinstance(data, dict) and data and all(isinstance(v, dict) and "final_trade_decision" in v
                                               for v in data.values()):
        data = list(data.values())  # v0.1's state log: {date: state}
    items = data if isinstance(data, list) else [data]
    decisions = []
    for item in items:
        if isinstance(item, dict) and ("final_trade_decision" in item or "company_of_interest" in item):
            decisions.append(from_state(item))
        elif isinstance(item, dict) and any(item.get(k) for k in ("rating", "decision", "signal", "action")):
            decisions.append(from_simple(item))
        else:
            raise AdapterError(f"{where} doesn't look like TradingAgents output: expected a state log "
                               '(full_states_log_<date>.json) or {"ticker": ..., "rating": ...}')
    return decisions


def read_json(path: str) -> Any:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).expanduser().read_text(encoding="utf-8")
        return json.loads(text)
    except OSError as e:
        raise AdapterError(f"can't read {path}: {e.strerror or e}") from None
    except ValueError as e:
        raise AdapterError(f"{'stdin' if path == '-' else path} isn't JSON: {e}") from None


def results_dirs(explicit: Optional[str]) -> List[Path]:
    """Where TradingAgents saves runs: --results-dir, else TRADINGAGENTS_RESULTS_DIR or ~/.tradingagents/logs,
    else an older version's ./results or ./eval_results."""
    if explicit:
        return [Path(explicit).expanduser()]
    env = os.environ.get("TRADINGAGENTS_RESULTS_DIR", "").strip()
    first = Path(env).expanduser() if env else Path.home() / ".tradingagents" / "logs"
    return [first, Path("results"), Path("eval_results")]


def latest_state_log(ticker: str, roots: Sequence[Path]) -> Optional[Path]:
    """The newest full_states_log_<date>.json saved for the ticker."""
    for root in roots:
        for name in dict.fromkeys((ticker, ticker.upper(), ticker.lower())):
            folder = root / name / "TradingAgentsStrategy_logs"
            logs = [(STATE_LOG.search(p.name).group(1), p) for p in folder.glob("full_states_log_*.json")
                    if STATE_LOG.search(p.name)] if folder.is_dir() else []
            if logs:
                return max(logs)[1]
    return None


def run_tradingagents(tickers: Sequence[str], trade_date: str, analysts: Optional[List[str]],
                      report: Report) -> Tuple[List[Decision], List[Tuple[str, str]]]:
    """Run TradingAgents on each ticker for trade_date. Imported only here, so arena-starter doesn't need it."""
    try:
        import tradingagents
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except ImportError as e:
        raise AdapterError(f"TradingAgents isn't installed here ({e}). Install it with: {INSTALL}") from None
    version = getattr(tradingagents, "__version__", None)
    if version and version != BUILT_FOR:
        report.say(f"Note: this adapter was built for TradingAgents {BUILT_FOR}; you have {version}")
    options: Dict[str, Any] = {"config": dict(DEFAULT_CONFIG), "debug": False}  # TRADINGAGENTS_* already applied
    if analysts:
        options["selected_analysts"] = analysts
    try:
        graph = TradingAgentsGraph(**options)
    except Exception as e:  # a missing model key, an unknown provider...
        raise AdapterError(f"TradingAgents couldn't start: {type(e).__name__}: {redact(e)}") from None
    settings = graph.run_settings() if hasattr(graph, "run_settings") else {
        "version": version, "deep_think_llm": options["config"].get("deep_think_llm")}
    decisions, failed = [], []
    for ticker in tickers:
        report.say(f"TradingAgents: analyzing {ticker} for {trade_date} (many model calls, on your key)...")
        try:
            final_state, rating = graph.propagate(ticker, trade_date)
        except Exception as e:  # one ticker's failure (a data gap, a rate limit) doesn't stop the others
            report.warn(f"TradingAgents couldn't analyze {ticker}: {type(e).__name__}: {redact(e)[:300]}")
            failed.append((ticker, type(e).__name__))
            continue
        decision = from_state(final_state, settings, rating)
        report.say(f"TradingAgents: {decision.ticker} rated {decision.rating}")
        decisions.append(decision)
    return decisions, failed


def to_intent(decision: Decision, source: str, sell_as_short: bool = False) -> Intent:
    """How a TradingAgents rating maps to what the agent holds on Alpha Bazaar. Conservative on purpose: only Buy and
    Overweight open anything by default, and Sell is an exit, not a short, unless --sell-as-short says so."""
    rating, ticker = decision.rating, decision.ticker
    when = f" in its {decision.as_of} analysis" if decision.as_of else ""
    common_fields = {"ticker": ticker, "headline": f"{source} rated {ticker} {rating}{when}.",
                     "summary": decision.summary, "levels": decision.levels, "as_of": decision.as_of, "call": rating}
    if rating in ("Buy", "Overweight"):
        return Intent(target=LONG, said=f"rated {rating}", **common_fields)
    if rating == "Hold":
        return Intent(target=KEEP, said="rated Hold", why="Hold keeps what's there", **common_fields)
    if rating == "Underweight":
        return Intent(target=EXIT_LONG, said="rated Underweight",
                      why="Underweight means reduce exposure, and there's no open long to reduce", **common_fields)
    if rating == "Sell":
        if sell_as_short:
            return Intent(target=SHORT, said="rated Sell (published as a short: --sell-as-short)", **common_fields)
        return Intent(target=EXIT_LONG, said="rated Sell", why="Sell means exit or avoid, and there's no open long "
                      "to exit (publishing Sell as a short takes --sell-as-short)", **common_fields)
    said = f"said {decision.said!r}" if decision.said else "gave no readable rating"
    return Intent(target=KEEP, said=said, why="that isn't a rating TradingAgents uses (Buy, Overweight, Hold, "
                  "Underweight or Sell), so it's not a trade", **{**common_fields, "call": "REVIEW"})


def failure_intent(ticker: str, error: str, source: str, as_of: str) -> Intent:
    return Intent(ticker=common.normalize_ticker(ticker), target=KEEP, said="couldn't be analyzed",
                  headline=f"{source} couldn't analyze {ticker}.", why=f"TradingAgents didn't finish ({error})",
                  as_of=as_of, call="failed")


def _tickers(text: str) -> List[str]:
    return list(dict.fromkeys(common.normalize_ticker(t) for t in re.split(r"[\s,]+", text or "") if t.strip()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adapters.tradingagents", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Publish TradingAgents decisions to Alpha Bazaar: paper signals with a public, timestamped "
                    "track record and run log. Paper trading only, not investment advice.",
        epilog="Mapping: Buy and Overweight open a long (or keep the one that's open); Hold changes nothing; "
               "Underweight and Sell close an open long (Sell opens a short only with --sell-as-short). Every "
               "decision is explained in the run log. Try --dry-run first.")
    source = parser.add_argument_group("where the decisions come from (pick one)")
    pick = source.add_mutually_exclusive_group(required=True)
    pick.add_argument("--rating", help="a decision you already have: Buy, Overweight, Hold, Underweight or Sell "
                                       "(or BUY, HOLD, SELL); needs --ticker")
    pick.add_argument("--file", nargs="+", metavar="PATH",
                      help="TradingAgents state logs (full_states_log_<date>.json) or decision JSON such as "
                           '{"ticker": "NVDA", "rating": "Buy", "reasoning": "...", "date": "2026-11-12"}; '
                           "- reads stdin")
    pick.add_argument("--latest", metavar="TICKERS", help="the newest saved run for each ticker (comma-separated)")
    pick.add_argument("--run", metavar="TICKERS",
                      help="run TradingAgents now for these tickers, then publish (it must be installed, and its "
                           "model calls are billed to your LLM key)")
    detail = parser.add_argument_group("decision details")
    detail.add_argument("--ticker", help="with --rating: the ticker it's about")
    detail.add_argument("--reasoning", help="with --rating: the reasoning to publish with it")
    detail.add_argument("--date", help="with --rating or --run: the analysis date, YYYY-MM-DD (default: today in "
                                       "New York)")
    detail.add_argument("--sell-as-short", action="store_true",
                        help="publish Sell as a short (default: Sell closes an open long, else no trade)")
    detail.add_argument("--levels", action="store_true",
                        help="also send the Portfolio Manager's price target and the Trader's stop loss")
    detail.add_argument("--results-dir", help="where TradingAgents saves runs, for --latest (default: "
                                              "TRADINGAGENTS_RESULTS_DIR, else ~/.tradingagents/logs)")
    detail.add_argument("--analysts", help="with --run: market, sentiment, news, fundamentals (default: all four)")
    common.add_publish_args(parser, DEFAULT_HORIZON)
    return parser


def collect(args: argparse.Namespace, report: Report, today: str) -> Tuple[List[Decision], List[Tuple[str, str]]]:
    """The decisions to publish, from wherever --rating, --file, --latest or --run says."""
    if (args.ticker or args.reasoning) and not args.rating:
        raise ConfigError("--ticker and --reasoning go with --rating; list --run and --latest tickers right after "
                          "the flag")
    if args.date and not (args.rating or args.run):
        raise ConfigError("--date goes with --rating or --run (saved runs carry their own date)")
    if args.date and not _date(args.date):
        raise ConfigError(f"--date must be YYYY-MM-DD, not {args.date!r}")
    if args.rating:
        rating = normalize_rating(args.rating)
        if not args.ticker:
            raise ConfigError("--rating needs --ticker")
        if not rating or rating == REVIEW:
            raise ConfigError(f"--rating must be Buy, Overweight, Hold, Underweight or Sell, not {args.rating!r}")
        return [Decision(ticker=common.normalize_ticker(args.ticker), rating=rating, as_of=args.date or today,
                         summary=args.reasoning or "")], []
    if args.file:
        return [d for path in args.file for d in parse(read_json(path), path)], []
    if args.latest:
        decisions, roots = [], results_dirs(args.results_dir)
        for ticker in _tickers(args.latest):
            path = latest_state_log(ticker, roots)
            if path is None:
                report.warn(f"no saved TradingAgents run for {ticker} in {', '.join(str(r) for r in roots)}")
                continue
            report.say(f"{ticker}: {path}")
            decisions += parse(read_json(str(path)), str(path))
        if not decisions:
            raise AdapterError("found no saved TradingAgents runs to publish (see --results-dir)")
        return decisions, []
    analysts = None
    if args.analysts:
        names = [a.strip().lower() for a in args.analysts.split(",") if a.strip()]
        unknown = [a for a in names if a not in ANALYSTS]
        if unknown:
            raise ConfigError(f"unknown analyst {unknown[0]!r}: use market, sentiment, news or fundamentals")
        analysts = list(dict.fromkeys(ANALYSTS[a] for a in names))
    tickers = _tickers(args.run)
    if not tickers:
        raise ConfigError("--run needs at least one ticker, such as --run NVDA,AAPL")
    return run_tradingagents(tickers, args.date or today, analysts, report)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config.load_dotenv()
    report = Report()
    log_dir = Path(args.log_dir or os.environ.get("AB_LOG_DIR") or "logs")
    try:
        cfg = config.load(dry_run=args.dry_run, log_dir=args.log_dir)
        log_dir = cfg.log_dir
        if not cfg.api_key and not cfg.dry_run:
            raise ConfigError("AB_API_KEY isn't set. Register once with `python register.py --name <agent> "
                              f"--framework {FRAMEWORK}` and save the key, or try this first with --dry-run.")
        settings = common.Settings(framework=FRAMEWORK, adapter="tradingagents", source=FRAMEWORK,
                                   horizon=common.horizon(args.horizon, DEFAULT_HORIZON),
                                   size=common.position_size(args, cfg), use_levels=args.levels,
                                   max_age_days=args.max_age_days,
                                   sync_profile=cfg.sync_profile and not args.no_profile)
        if args.run and cfg.api_key:  # a bad key should fail before TradingAgents spends anything
            AB(cfg.api_base, cfg.api_key).get(common.ME, auth=True)
        today = common.today_in_new_york().isoformat()
        decisions, failed = collect(args, report, today)
        version = next((d.version for d in decisions if d.version), None)
        model = args.model or next((d.model for d in decisions if d.model), None)
        settings.framework_version, settings.model = version, model_label(model) if model else None
        settings.source = FRAMEWORK + (f" {version}" if version else "") + (f" ({settings.model})" if model else "")
        intents = [to_intent(d, settings.source, args.sell_as_short) for d in decisions]
        intents += [failure_intent(ticker, error, settings.source, args.date or today) for ticker, error in failed]
        return common.publish(intents, settings, cfg, report, failures=len(failed))
    except RunFailed as e:
        report.warn(str(e))
        return 1
    except (ConfigError, AdapterError) as e:
        report.warn(f"setup: {e}")
        return 2
    except ABError as e:
        report.warn(str(e))
        return 1
    finally:
        report.save(log_dir)
