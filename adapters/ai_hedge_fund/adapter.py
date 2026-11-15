"""Publish ai-hedge-fund decisions to Alpha Bazaar as paper signals, with a public run log.

  aihf paper tick alpha | python -m adapters.ai_hedge_fund --file -   # what a paper fund just decided
  python -m adapters.ai_hedge_fund --fund alpha                       # the newest record in its ledger
  python -m adapters.ai_hedge_fund --run --universe AAPL,MSFT,NVDA    # run a fund now (installed separately)
  python -m adapters.ai_hedge_fund --file decisions.json              # v1's {"decisions": {...}} output
  python -m adapters.ai_hedge_fund --ticker NVDA --action buy --quantity 25 --confidence 80 --reasoning "..."

Built against ai-hedge-fund 2.7.2 (pip install aihf; virattt/ai-hedge-fund@5c553b5, 2026-11-09). Since 2.0 a fund
decides target weights: `aihf paper tick NAME` advances a deployed paper fund one session, appends a SessionRecord to
~/.hedge-fund/paper/NAME/ledger/ and prints it as JSON. Its `decision` (made at that close, executed at the next)
carries final_weights (signed fractions of NAV, after risk limits) and each analyst's view: a conviction from -1 to
+1 with its reasoning. `aihf backtest` prints {"records": [...]} of the same records.

The v1 app (src/main.py, until release 2026.7.10) decided per ticker instead: {"decisions": {"NVDA": {"action": "buy",
"quantity": 25, "confidence": 80, "reasoning": "..."}}}, with buy, sell, short, cover or hold, from run_hedge_fund().
Both shapes work here.

How they map to Alpha Bazaar (see the README): a positive weight or buy opens a long, a negative weight or short
opens a short, a zero weight closes what's open, sell closes a long and cover a short, and hold changes nothing.
Alpha Bazaar closes a signal in full: a partial sell closes the whole long.
"""
import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adapters import common
from adapters.common import EXIT_LONG, EXIT_SHORT, FLAT, KEEP, LONG, SHORT, AdapterError, Intent, RunFailed
from agent import config
from agent.ab import AB, ABError, redact
from agent.config import ConfigError
from agent.main import Report
from agent.providers import model_label

FRAMEWORK = "ai-hedge-fund"
BUILT_FOR = "2.7.2"
DEFAULT_HORIZON = "1M"  # a fund holds a name until a rebalance changes it; a later run closes the signal sooner
DEFAULT_PORTFOLIO_VALUE = 100_000.0  # ai-hedge-fund's starting cash, for v1 sizing hints
QUANT_LABEL = "quant models (no LLM)"
INSTALL = "pip install aihf==2.7.2"
PAPER_DIR = Path.home() / ".hedge-fund" / "paper"
WEIGHT_EPSILON = 1e-6
VIEWS_SHOWN = 4
V1_ACTIONS = {"buy": LONG, "short": SHORT, "sell": EXIT_LONG, "cover": EXIT_SHORT, "hold": KEEP}
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_BAR = re.compile(r"\d{6}")


@dataclass
class Parsed:
    """ai-hedge-fund output, as intents plus what the run log should say about it."""

    intents: List[Intent] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    source: str = FRAMEWORK  # who decided, as the run log names it: "ai-hedge-fund 2.7.2 (claude-opus-5-5)"
    version: Optional[str] = None
    model: Optional[str] = None  # the model label for the profile
    fund: Optional[str] = None


# --- v2: a fund's session record (aihf paper tick, the ledger, aihf backtest) ---------------------------------------

def _views(basis: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
    """Each ticker's analyst views, and the fund's net conviction in it (strategy slices times blended views)."""
    views: Dict[str, List[Dict[str, Any]]] = {}
    net: Dict[str, float] = {}
    for strategy in basis.get("strategies") or []:
        share = common.number(strategy.get("slice")) or 0.0
        for ticker, conviction in (strategy.get("convictions") or {}).items():
            value = common.number(conviction)
            if value is not None:
                key = common.normalize_ticker(ticker)
                net[key] = net.get(key, 0.0) + share * value
        for signal in strategy.get("signals") or []:
            views.setdefault(common.normalize_ticker(signal.get("ticker")), []).append(signal)
    return views, net


def _views_text(views: Sequence[Dict[str, Any]]) -> str:
    strongest = sorted(views, key=lambda s: -abs(common.number(s.get("value")) or 0.0))[:VIEWS_SHOWN]
    parts = []
    for view in strongest:
        value = common.number(view.get("value"))
        reasoning = common.summarize(view.get("reasoning"), 280)
        parts.append(f"{view.get('model_name', 'analyst')} {value:+.2f}" + (f": {reasoning}" if reasoning else "")
                     if value is not None else str(view.get("model_name", "analyst")))
    return " · ".join(parts)


def source_label(version: Optional[str], model: Optional[str]) -> str:
    """"ai-hedge-fund 2.7.2 (claude-opus-5-5)": the release without a checkout's commit, and the model."""
    release = (version or "").split("+")[0]
    return FRAMEWORK + (f" {release}" if release else "") + (f" ({model})" if model else "")


def from_record(record: Dict[str, Any], model: Optional[str] = None) -> Parsed:
    """Intents from one SessionRecord: its new decision, else the decision it executed, else nothing new.
    `model` (--model) replaces the record's own llm_model on the profile."""
    fund, session = record.get("fund"), str(record.get("session") or "")
    label = model or (model_label(record["llm_model"]) if record.get("llm_model") else QUANT_LABEL)
    parsed = Parsed(version=record.get("code_version"), fund=fund, model=label,
                    source=source_label(record.get("code_version"), label))
    source = parsed.source
    positions = {common.normalize_ticker(t): int(common.number(s) or 0) for t, s in
                 (record.get("positions") or {}).items()}
    basis = record.get("decision")
    if not basis and record.get("executed"):
        basis = record["executed"]
        parsed.notes.append(f"This session ({session}) made no new decision; it executed the one made at the "
                            f"{basis.get('as_of')} close, so that's the one published.")
    if not basis:
        held = ", ".join(f"{t} {s:+d}" for t, s in sorted(positions.items()) if s) or "nothing"
        parsed.notes.append(f"No rebalance at the {session} session, so there's no new decision to publish. The fund "
                            f"holds {held}.")
        return parsed
    as_of = str(basis.get("as_of") or session)[:10]
    weights = {common.normalize_ticker(t): common.number(w) or 0.0 for t, w in
               (basis.get("final_weights") or {}).items()}
    skipped = {common.normalize_ticker(s.get("ticker")): s.get("reason") for s in basis.get("skipped") or []}
    stopped = any(c.get("limit") == "max_drawdown_pct" for c in basis.get("clamps") or [])
    if stopped:
        parsed.notes.append("Drawdown stop-out: the fund sells everything and stays in cash.")
    views, net = _views(basis)
    universe = [common.normalize_ticker(t) for t in basis.get("universe") or record.get("universe") or []]
    for ticker in dict.fromkeys([*universe, *weights, *positions]):
        held = positions.get(ticker, 0)
        holding = f", holds {held:+d} shares" if held else ""
        if ticker in skipped:
            parsed.intents.append(Intent(ticker=ticker, target=KEEP, said="was skipped by the fund", headline="",
                                         why=f"ai-hedge-fund skipped it: {skipped[ticker]}", as_of=as_of,
                                         call="skipped"))
            continue
        weight = weights.get(ticker, 0.0)
        target = LONG if weight > WEIGHT_EPSILON else SHORT if weight < -WEIGHT_EPSILON else FLAT
        summary = _views_text(views.get(ticker, []))
        if ticker in net:
            summary = f"Net conviction {net[ticker]:+.2f}. {summary}".strip()
        parsed.intents.append(Intent(
            ticker=ticker, target=target, said=f"target {weight:+.1%} of NAV{holding}",
            headline=f"{source}: fund {fund} targets {ticker} at {weight:+.1%} of NAV after the {as_of} close.",
            summary=("Drawdown stop-out. " if stopped else "") + summary,
            why="no position held, none wanted" if not held else "nothing open on Alpha Bazaar to close",
            confidence=common.confidence_from(net[ticker], 1.0) if ticker in net and target != FLAT else None,
            size_hint=abs(weight) * 100 or None, as_of=as_of, call=f"{weight:+.1%}",
            minor=target == FLAT and not held))
    return parsed


# --- v1: per-ticker decisions (run_hedge_fund) and decisions given on the command line --------------------------------

def _analysts(signals: Dict[str, Any], ticker: str) -> str:
    """"Analysts: 4 bullish, 1 bearish, 2 neutral." from v1's analyst_signals."""
    tally: Dict[str, int] = {}
    for agent, by_ticker in (signals or {}).items():
        signal = (by_ticker or {}).get(ticker) if isinstance(by_ticker, dict) else None
        if agent.startswith("risk_management") or not isinstance(signal, dict) or not signal.get("signal"):
            continue
        word = str(signal["signal"]).lower()
        tally[word] = tally.get(word, 0) + 1
    return ("Analysts: " + ", ".join(f"{n} {word}" for word, n in sorted(tally.items(), key=lambda i: -i[1])) + ".") \
        if tally else ""


def v1_intent(ticker: str, decision: Dict[str, Any], source: str, as_of: str, price: Optional[float] = None,
              portfolio_value: float = DEFAULT_PORTFOLIO_VALUE, analysts: str = "") -> Intent:
    ticker = common.normalize_ticker(ticker)
    action = str(decision.get("action") or "").strip().lower()
    quantity = common.number(decision.get("quantity"))
    confidence = common.number(decision.get("confidence"))
    target = V1_ACTIONS.get(action, KEEP)
    shares = f" {quantity:g} shares" if quantity is not None else ""
    conviction = f", confidence {confidence:g}/100" if confidence is not None else ""
    why = {KEEP: "hold keeps what's there", EXIT_LONG: "there's no open long to sell",
           EXIT_SHORT: "there's no open short to cover"}.get(target, "")
    if action not in V1_ACTIONS:
        why = f"{action or 'no action'!r} isn't one of ai-hedge-fund's actions (buy, sell, short, cover, hold)"
    elif quantity == 0 and target != KEEP:
        target, why = KEEP, "0 shares is no trade"
    summary = " ".join(part for part in (common.summarize(decision.get("reasoning"), 1000), analysts) if part)
    with_confidence = f" with confidence {confidence:g}/100" if confidence is not None else ""
    return Intent(ticker=ticker, target=target, said=f"decided {action or '?'}{shares}{conviction}",
                  headline=f"{source} decided to {action}{shares}{' of' if shares else ''} {ticker}{with_confidence}.",
                  summary=summary, why=why, confidence=common.confidence_from(confidence, 100.0),
                  size_hint=quantity * price / portfolio_value * 100 if quantity and price else None,
                  as_of=as_of, call=f"{action} {quantity:g}" if quantity is not None else action)


def from_v1(data: Dict[str, Any], model: Optional[str], as_of: str, portfolio_value: float) -> Parsed:
    decisions = data.get("decisions") if "decisions" in data else data
    if not isinstance(decisions, dict) or not decisions:
        raise AdapterError("the v1 output has no decisions in it (run_hedge_fund returns None for decisions when "
                           "the portfolio manager's answer wasn't JSON)")
    source = source_label(None, model)
    signals = data.get("analyst_signals") or {}
    prices = data.get("current_prices") or {}
    risk = signals.get("risk_management_agent") or {}
    intents = []
    for ticker, decision in decisions.items():
        if not isinstance(decision, dict):
            raise AdapterError(f"the decision for {ticker} isn't an object: {str(decision)[:100]}")
        price = common.number(prices.get(ticker)) or common.number((risk.get(ticker) or {}).get("current_price"))
        intents.append(v1_intent(ticker, decision, source, as_of, price, portfolio_value,
                                 _analysts(signals, ticker)))
    return Parsed(intents=intents, source=source, model=model)


# --- reading output -------------------------------------------------------------------------------------------------

def parse(data: Any, where: str, model: Optional[str], as_of: str, portfolio_value: float) -> Parsed:
    """Parsed output from any of ai-hedge-fund's JSON shapes. `model` is --model's label, if given."""
    if isinstance(data, dict) and isinstance(data.get("records"), list):  # aihf backtest: publish its last session
        if not data["records"]:
            raise AdapterError(f"{where} is a backtest with no sessions in it")
        data = data["records"][-1]
    if isinstance(data, dict) and "session" in data and ("decision" in data or "positions" in data):
        return from_record(data, model)
    if isinstance(data, dict) and (isinstance(data.get("decisions"), dict) or "decisions" in data):
        return from_v1(data, model, as_of, portfolio_value)
    if isinstance(data, dict) and data and all(isinstance(v, dict) and "action" in v for v in data.values()):
        return from_v1({"decisions": data}, model, as_of, portfolio_value)
    raise AdapterError(f"{where} doesn't look like ai-hedge-fund output: expected `aihf paper tick` or `aihf "
                       'backtest` JSON, a ledger record, or v1\'s {"decisions": {...}}')


def read_json(path: str) -> Any:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).expanduser().read_text(encoding="utf-8")
        return json.loads(text)
    except OSError as e:
        raise AdapterError(f"can't read {path}: {e.strerror or e}") from None
    except ValueError as e:
        raise AdapterError(f"{'stdin' if path == '-' else path} isn't JSON: {e}") from None


def latest_record(fund_dir: Path) -> Path:
    """The newest session record in a paper fund's ledger: ledger/<date>.json, or ledger/<date>/<HHMMSS>.json for a
    minute fund."""
    ledger = fund_dir / "ledger"
    if not ledger.is_dir():
        raise AdapterError(f"{fund_dir} has no ledger: create the fund with `aihf paper create` and run "
                           "`aihf paper tick` first (or see --paper-dir)")
    found = []
    for entry in ledger.iterdir():
        if entry.is_file() and entry.suffix == ".json" and _DAY.fullmatch(entry.stem):
            found.append((entry.stem, entry))
        elif entry.is_dir() and _DAY.fullmatch(entry.name):
            found += [(f"{entry.name}T{bar.stem}", bar) for bar in entry.glob("*.json") if _BAR.fullmatch(bar.stem)]
    if not found:
        raise AdapterError(f"{fund_dir}'s ledger has no sessions yet: run `aihf paper tick {fund_dir.name}` first")
    return max(found)[1]


def run_fund(fund: Optional[str], mandate: Optional[str], universe: List[str], paper_dir: Path,
             report: Report) -> Optional[Dict[str, Any]]:
    """Advance a paper fund one session with ai-hedge-fund itself, and return the record (None if none is due).
    With a fund name, that deployed fund ticks, as `aihf paper tick` does. Without one, a fresh fund from the mandate
    is deployed in a temporary folder and ticks once: its decision is the book it would hold today (what CI does,
    since a runner keeps no ledger between runs). Imported only here, so arena-starter doesn't need ai-hedge-fund."""
    try:
        from hedge_fund.data import CachedDataClient, FDClient
        from hedge_fund.fund import load_spec
        from hedge_fund.paper import NothingDue, deploy, load_deployed, tick, validate_fund_name
        from hedge_fund.paths import EXAMPLE_MANDATE
        from hedge_fund.pipeline import FundHalted
        from hedge_fund.tui.keys import apply_credentials
    except ImportError as e:
        raise AdapterError(f"ai-hedge-fund isn't installed here ({e}). Install it with: {INSTALL}") from None
    apply_credentials()  # its keys from ./.env and ~/.hedge-fund/.env; the environment wins
    if fund:
        directory = paper_dir / fund
        if not (directory / "fund.yaml").is_file():
            raise AdapterError(f"no paper fund {fund!r} in {paper_dir}: create it with `aihf paper create {fund} "
                               "--mandate ... --universe ...`")
    else:
        try:
            spec = load_spec(mandate or EXAMPLE_MANDATE)
        except (OSError, ValueError) as e:
            raise AdapterError(f"can't use the mandate {mandate or EXAMPLE_MANDATE}: {e}") from None
        try:
            name = validate_fund_name(spec.name)
        except ValueError:
            name = "arena-starter"
        directory = deploy(name, spec, universe, root=Path(tempfile.mkdtemp(prefix="aihf-")))
        report.say(f"ai-hedge-fund: a fresh fund from the mandate {spec.name} over {', '.join(universe)}")
    deployed = load_deployed(directory)
    report.say(f"ai-hedge-fund: advancing {deployed.name} one session (its model calls are billed to your keys)...")
    try:
        with FDClient() as raw:
            record = tick(directory, CachedDataClient(raw, live=deployed.spec.is_intraday))
    except NothingDue as e:
        report.say(f"ai-hedge-fund: {e}")
        return None
    except FundHalted as e:
        raise AdapterError(f"{e} (`aihf paper resume {deployed.name}` clears the kill switch)") from None
    except Exception as e:  # ai-hedge-fund halts a fund whose tick fails; the reason is in its events.jsonl
        raise RunFailed(f"ai-hedge-fund's tick failed: {type(e).__name__}: {redact(e)[:300]}") from None
    return json.loads(record.model_dump_json())


# --- command line ---------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adapters.ai_hedge_fund", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Publish ai-hedge-fund decisions to Alpha Bazaar: paper signals with a public, timestamped "
                    "track record and run log. Paper trading only, not investment advice.",
        epilog="Mapping: a positive target weight (or buy) opens a long, a negative one (or short) opens a short, a "
               "zero weight closes what's open, sell closes a long, cover closes a short, hold changes nothing. "
               "Every decision is explained in the run log. Try --dry-run first.")
    source = parser.add_argument_group("where the decisions come from (pick one)")
    source.add_argument("--file", metavar="PATH", help="`aihf paper tick` or `aihf backtest` JSON, a ledger record, "
                                                       "or v1's decisions JSON; - reads stdin")
    source.add_argument("--fund", metavar="NAME", help="a deployed paper fund: its newest ledger record (with --run: "
                                                       "tick it once first)")
    source.add_argument("--action", help="one decision you already have: buy, sell, short, cover or hold; needs "
                                         "--ticker")
    source.add_argument("--run", action="store_true",
                        help="run ai-hedge-fund now, then publish: tick --fund NAME, or a fresh fund from --mandate "
                             "over --universe (it must be installed, and its data and model calls are billed to "
                             "your keys)")
    detail = parser.add_argument_group("decision details")
    detail.add_argument("--ticker", help="with --action: the ticker")
    detail.add_argument("--quantity", type=float, help="with --action: shares (0 means no trade)")
    detail.add_argument("--confidence", type=float, help="with --action: confidence, 0-100")
    detail.add_argument("--reasoning", help="with --action: the reasoning to publish with it")
    detail.add_argument("--date", help="with --action or v1 JSON: the date it was decided, YYYY-MM-DD (default: "
                                       "today in New York)")
    detail.add_argument("--universe", help="with --run and no --fund: the tickers the fresh fund trades")
    detail.add_argument("--mandate", help="with --run and no --fund: the mandate YAML (default: ai-hedge-fund's "
                                          "example mandate)")
    detail.add_argument("--paper-dir", help=f"where paper funds live (default: {PAPER_DIR})")
    detail.add_argument("--size-from-fund", action="store_true",
                        help="size each new signal from the fund's own sizing: its target weight, or v1's quantity "
                             "times price over --portfolio-value (default: --size for every signal)")
    detail.add_argument("--portfolio-value", type=float, default=DEFAULT_PORTFOLIO_VALUE,
                        help=f"for v1 sizing: the fund's value in dollars (default: {DEFAULT_PORTFOLIO_VALUE:,.0f}, "
                             "ai-hedge-fund's starting cash)")
    common.add_publish_args(parser, DEFAULT_HORIZON)
    return parser


def _universe(text: Optional[str]) -> List[str]:
    return list(dict.fromkeys(common.normalize_ticker(t) for t in re.split(r"[\s,]+", text or "") if t.strip()))


def collect(args: argparse.Namespace, report: Report, today: str) -> Parsed:
    """The decisions to publish, from wherever --file, --fund, --action or --run says."""
    sources = [bool(args.file), bool(args.action), bool(args.fund or args.run)]
    if sum(sources) != 1:
        raise ConfigError("pick one source: --file, --fund, --action or --run")
    if (args.ticker or args.quantity is not None or args.confidence is not None or args.reasoning) and not args.action:
        raise ConfigError("--ticker, --quantity, --confidence and --reasoning go with --action")
    if (args.universe or args.mandate) and not (args.run and not args.fund):
        raise ConfigError("--universe and --mandate go with --run (for a fresh fund; a deployed fund has its own)")
    if args.date and not _DAY.fullmatch(args.date):
        raise ConfigError(f"--date must be YYYY-MM-DD, not {args.date!r}")
    if not args.portfolio_value > 0:
        raise ConfigError("--portfolio-value must be more than 0")
    as_of = args.date or today
    model = model_label(args.model) if args.model else None
    if args.action:
        if not args.ticker:
            raise ConfigError("--action needs --ticker")
        if args.action.strip().lower() not in V1_ACTIONS:
            raise ConfigError(f"--action must be buy, sell, short, cover or hold, not {args.action!r}")
        decision = {"action": args.action, "quantity": args.quantity, "confidence": args.confidence,
                    "reasoning": args.reasoning}
        source = source_label(None, model)
        return Parsed(intents=[v1_intent(args.ticker, decision, source, as_of)], source=source, model=model)
    if args.file:
        return parse(read_json(args.file), "stdin" if args.file == "-" else args.file, model, as_of,
                     args.portfolio_value)
    paper_dir = Path(args.paper_dir).expanduser() if args.paper_dir else PAPER_DIR
    if args.run:
        universe = _universe(args.universe)
        if not args.fund and not universe:
            raise ConfigError("--run needs --fund NAME, or --universe TICKERS for a fresh fund")
        record = run_fund(args.fund, args.mandate, universe, paper_dir, report)
        if record is None:
            return Parsed(source=source_label(None, model), model=model)
        return parse(record, "the fund's new record", model, as_of, args.portfolio_value)
    path = latest_record(paper_dir / args.fund)
    report.say(f"{args.fund}: {path}")
    return parse(read_json(str(path)), str(path), model, as_of, args.portfolio_value)


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
        settings = common.Settings(framework=FRAMEWORK, adapter="ai-hedge-fund", source=FRAMEWORK,
                                   horizon=common.horizon(args.horizon, DEFAULT_HORIZON),
                                   size=common.position_size(args, cfg), size_from_framework=args.size_from_fund,
                                   max_age_days=args.max_age_days,
                                   sync_profile=cfg.sync_profile and not args.no_profile)
        if args.run and cfg.api_key:  # a bad key should fail before the fund spends anything
            AB(cfg.api_base, cfg.api_key).get(common.ME, auth=True)
        parsed = collect(args, report, common.today_in_new_york().isoformat())
        settings.framework_version, settings.model = parsed.version, parsed.model
        settings.source = parsed.source + (f", fund {parsed.fund}" if parsed.fund else "")
        return common.publish(parsed.intents, settings, cfg, report, notes=parsed.notes)
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
