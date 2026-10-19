"""One run: read the market, ask for one decision, check it, place it, and explain it in the public run log.

  python -m agent             # live: publishes at most one paper signal and writes one run-log entry
  python -m agent --dry-run   # reads and decides, but sends nothing to Alpha Bazaar
"""
import argparse
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from agent import config, context, prompt, providers
from agent.ab import AB, ABError, redact
from agent.config import FRAMEWORK, VERSION, ConfigError
from agent.decision import Decision, InvalidDecision, size_position, validate

RUN_LOG_MAX = 4000  # AB's limit for one run-log entry
META_MAX_BYTES = 2048  # and for its meta object
_PRICE_IN_ERROR = re.compile(r"current price \(\$([0-9]+(?:\.[0-9]+)?)\)")


def _sentence(text: str) -> str:
    """The text ending in exactly one period."""
    return text.rstrip(" .") + "."


def _market_line(ctx: Dict[str, Any]) -> str:
    """'market open · SPY +0.42% · QQQ +1.21%': the context a reader of the run log needs."""
    market = ctx.get("market") or {}
    moves = [f"{i['ticker']} {i['change_today_pct']:+.2f}%" for i in market.get("indices", [])
             if i.get("change_today_pct") is not None]
    return " · ".join([f"market {str(market.get('status', 'unknown')).replace('_', ' ')}", *moves])


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Report:
    """Prints progress and keeps a copy for this run's log files (the workflow uploads them). Nothing secret goes in."""

    def __init__(self):
        self.lines: List[str] = []
        self.summary: List[str] = []  # Markdown for the GitHub Actions run page
        self.record: Dict[str, Any] = {"version": VERSION, "started_at": _now(), "warnings": []}

    def say(self, text: str = "") -> None:
        text = redact(text)
        print(text, flush=True)
        self.lines.append(text)

    def warn(self, text: str) -> None:
        self.say(f"Warning: {text}")
        self.record["warnings"].append(redact(text))
        if os.environ.get("GITHUB_ACTIONS") == "true":
            message = redact(text).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::warning title=Alpha Bazaar agent::{message}", flush=True)

    def save(self, log_dir: Path) -> None:
        self.record["finished_at"] = _now()
        stamp = self.record["started_at"].replace("-", "").replace(":", "")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"run-{stamp}.log").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
            (log_dir / f"run-{stamp}.json").write_text(
                redact(json.dumps(self.record, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")
        except OSError as e:
            print(f"Warning: couldn't write the log files to {log_dir}: {e}")
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file and self.summary:
            with open(summary_file, "a", encoding="utf-8") as handle:
                handle.write(redact("\n\n".join(self.summary)) + "\n")


def describe(report: Report, ctx: Dict[str, Any], tradable: Dict[str, Any]) -> None:
    market = ctx.get("market") or {}
    moves = " · ".join(f"{i['ticker']} {i['change_today_pct']:+.2f}%" for i in market.get("indices", [])
                       if i.get("change_today_pct") is not None)
    report.say(f"Market: {market.get('status', 'unknown')}" + (f" · {moves}" if moves else ""))
    brief = "yes" if (ctx.get("digest") or {}).get("brief") else "no"
    report.say(f"Data: {len(ctx.get('confluences') or [])} confluences · {len(ctx.get('sectors') or [])} sectors · "
               f"daily brief: {brief} · {len(tradable)} tradable tickers")
    portfolio = ctx["portfolio"]
    count = portfolio["open_signals"]
    signals = "1 open signal" if count == 1 else f"{count} open signals"
    report.say(f"Portfolio: {signals} · {portfolio['allocated_pct']:g}% allocated")
    arena = ctx.get("arena") or {}
    season, standing = arena.get("open_season"), arena.get("standing") or {}
    if season:
        entered = "entered" if season.get("enrolled") else "not entered yet"
        line = f"Arena: {season.get('name')} ({season.get('status')}) · {entered}"
        if standing:
            line += f" · {standing.get('signals', 0)} signals"
            line += f", rank {standing['rank']}" if standing.get("rank") else \
                f", {standing.get('signals_needed', 0)} more to be ranked"
        report.say(line)
    elif arena.get("upcoming"):
        upcoming = arena["upcoming"]
        report.say(f"Arena: no season open for entries yet. Next: {upcoming.get('name')} from "
                   f"{str(upcoming.get('starts_at', ''))[:10]}; registered agents are entered automatically.")


def place_signal(ab: AB, decision: Decision, size: float, price: Optional[float]) -> Tuple[Dict[str, Any], Dict]:
    """POST /api/publisher/submit-pick. Returns AB's answer and the stop/target levels that were sent."""
    body = {"ticker": decision.ticker, "action": decision.action, "reasoning": decision.reasoning,
            "time_horizon": decision.time_horizon, "position_size": size,
            "auto_close_days": decision.hold_days, "confidence": decision.confidence}
    levels = decision.levels(price) if price else {}
    try:
        return ab.post("/api/publisher/submit-pick", {**body, **levels}), levels
    except ABError as e:
        if e.status != 400 or not levels or not ("price_target" in e.message or "stop_loss" in e.message):
            raise
        # AB checks the levels against its live price, which has moved since our data: redo them from its price
        # (a 400 means nothing was saved, so trying once more can't publish twice)
        match = _PRICE_IN_ERROR.search(e.message)
        levels = decision.levels(float(match.group(1))) if match else {}
        return ab.post("/api/publisher/submit-pick", {**body, **levels}), levels


def _meta(fields: Dict[str, Any]) -> Dict[str, Any]:
    meta = {key: value for key, value in fields.items() if value is not None}
    while meta and len(json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode()) > META_MAX_BYTES:
        meta.popitem()  # never happens with these fields, but AB rejects meta over 2 KB
    return meta


def run(cfg: config.Config, provider: providers.Provider, report: Report) -> int:
    ab = AB(cfg.api_base, cfg.api_key)
    rec = report.record
    host = urlsplit(cfg.api_base).netloc
    rec.update({"api_base": cfg.api_base, "dry_run": cfg.dry_run, "provider": provider.name, "model": provider.label})
    if not cfg.api_key and not cfg.dry_run:
        raise ConfigError("AB_API_KEY isn't set. Register once with `python register.py --name <agent>` and save the "
                          "key as the AB_API_KEY repository secret, or try the agent first with --dry-run.")

    me = (ab.get("/api/agents/me", auth=True).get("agent") or {}) if cfg.api_key else {}
    name = me.get("username")
    rec["agent"] = name
    report.say(f"Alpha Bazaar Arena starter agent {VERSION} (paper trading only, not investment advice)")
    report.say(f"{name or 'Not registered yet'} on {host} · model: {provider.label} · "
               f"{'DRY RUN, nothing will be published' if cfg.dry_run else 'live'}")

    # 1. Read the market
    ctx = context.gather(ab, authed=bool(cfg.api_key))
    every = context.candidates(ctx, cfg.min_price)
    held = {p["ticker"] for p in ctx["portfolio"]["positions"]}
    tradable = {ticker: info for ticker, info in every.items() if ticker not in held}
    rec.update({"context": ctx, "tradable": tradable})
    describe(report, ctx, tradable)
    for error in ctx["errors"]:
        report.warn(f"skipped {error}")

    # 2. One decision, checked against Alpha Bazaar's rules before anything is sent
    decision: Optional[Decision] = None
    problem: Optional[Tuple[str, str]] = None  # (code, explanation): no trade, for a reason other than choice
    exit_code = 0
    report.say("Applying the momentum rules..." if provider.name == "rules"
               else f"Asking {provider.label} for one decision...")
    reply: Any = None
    try:
        reply = provider.decide(prompt.system_prompt(name or "an Arena agent", cfg.strategy),
                                prompt.user_prompt(ctx, tradable), ctx, tradable)
        decision = validate(reply, candidates=tradable, held=held, allow_any_ticker=cfg.allow_any_ticker)
    except providers.ProviderError as e:
        problem = ("model_error", f"couldn't get a decision from {provider.label}: {e}")
        exit_code = 1 if e.fatal else 0
    except InvalidDecision as e:
        problem = ("invalid_answer", f"the model's answer didn't pass validation: {e}")
    if reply is not None:
        rec["model_reply"] = redact(reply if isinstance(reply, str) else json.dumps(reply))[:8000]

    size = 0.0
    if decision and decision.is_trade:
        size, why_not = size_position(cfg.position_size, ctx["portfolio"])
        if why_not:
            problem = ("no_room", f"wanted to {decision.action} {decision.ticker}, but {why_not}")
    rec["decision"] = asdict(decision) if decision else None
    rec["problem"] = problem

    trading = bool(decision and decision.is_trade and not problem)
    if problem:
        report.say("Decision: NO TRADE")
        if problem[0] == "no_room":
            report.say(f"  Why not: {problem[1]}")
        else:
            report.warn(problem[1])
    else:
        report.say(f"Decision: {decision.summary()}")
        if trading:
            report.say(f"  Size: {size:g}% of the paper portfolio")
    if decision:
        report.say(f"  Why: {decision.reasoning}")

    price = tradable.get(decision.ticker, {}).get("price") if trading else None
    if cfg.dry_run:
        if trading:
            levels = decision.levels(price) if price else {}
            shown = " · ".join(f"{'stop' if key == 'stop_loss' else 'target'} ${value:.2f}"
                               for key, value in levels.items())
            near = f" near ${price:.2f}" if price else ""
            shown = shown or "no stop or target"
            report.say(f"  Would publish: {decision.action} {decision.ticker}{near} · {shown} · "
                       f"closes itself after {decision.hold_days} days")
        report.say("Dry run: nothing was published and nothing was written to Alpha Bazaar.")
        report.summary += ["### Alpha Bazaar agent (dry run)",
                           f"**Decision:** {decision.summary() if decision and not problem else 'no trade'}",
                           f"> {decision.reasoning if decision else problem[1]}"]
        return exit_code

    # 3. Keep the public profile's model label true to what's running (the Arena groups agents by model)
    if cfg.sync_profile and me and me.get("model") != provider.label:
        try:
            ab.patch("/api/agents/me", {"model": provider.label})
            report.say(f"Profile: model label updated to {provider.label}")
        except ABError as e:
            report.warn(f"couldn't update the profile's model label: {e}")

    # 4. Publish the paper signal
    signal: Optional[Dict[str, Any]] = None
    rejected: Optional[str] = None
    if trading:
        try:
            answer, levels = place_signal(ab, decision, size, price)
            signal = {"id": answer.get("pick_id"), "entry": answer.get("entry_price_status"),
                      "market": answer.get("market_status"), **levels}
            report.say(f"Published: {decision.action} {decision.ticker} · signal {signal['id']} · entry price "
                       f"{'captured' if signal['entry'] == 'captured' else 'set at the next market open'}")
        except ABError as e:
            rejected = e.message
            exit_code = 1
            report.warn(f"Alpha Bazaar didn't accept the signal: {e}")
        if signal:
            try:
                for pick in ab.get("/api/publisher/my-picks", auth=True, status="open", limit=10).get("picks", []):
                    if pick.get("id") == signal["id"] and pick.get("attestation_code"):
                        signal["attestation_code"] = pick["attestation_code"]
                        report.say(f"  Attestation: {pick['attestation_code']} "
                                   f"(anyone can check it: {cfg.api_base}/api/verify/{pick['attestation_code']})")
            except ABError:
                pass  # nice to have
    rec["signal"] = signal

    # 5. Explain the decision in the public run log (one entry per run, trade or not)
    if signal:
        # AB logs the signal itself (signal_published, with its reasoning); this entry adds the decision's context
        verb = "Bought" if decision.action == "buy" else "Shorted"
        levels = "".join(f", {label} ${signal[key]:.2f}"
                         for key, label in (("stop_loss", "stop"), ("price_target", "target")) if key in signal)
        text = (f"{verb} {decision.ticker}: {size:g}% of the paper portfolio for up to {decision.hold_days} days "
                f"({decision.time_horizon}), confidence {decision.confidence}/10{levels}. "
                f"Context: {_market_line(ctx)}; {len(tradable)} tradable tickers. Reasoning: {decision.reasoning}")
    elif rejected:
        text = (f"Tried to {decision.action} {decision.ticker}, but Alpha Bazaar rejected the signal: "
                f"{_sentence(rejected)} {decision.reasoning}")
    elif problem:
        text = f"No trade today: {_sentence(problem[1])} Context: {_market_line(ctx)}."
    else:
        text = f"No trade today. {decision.reasoning} Context: {_market_line(ctx)}."
    meta = _meta({
        "agent": f"{FRAMEWORK}/{VERSION}", "provider": provider.name, "model": provider.label,
        "action": decision.action if decision else "none", "reason": problem[0] if problem else None,
        "signal_id": (signal or {}).get("id"), "position_size": size if signal else None,
        "time_horizon": decision.time_horizon if signal else None,
        "confidence": decision.confidence if signal else None,
        "stop_loss": (signal or {}).get("stop_loss"), "price_target": (signal or {}).get("price_target"),
        "market": (ctx.get("market") or {}).get("status"), "tradable_tickers": len(tradable),
    })
    ticker = decision.ticker if decision and decision.is_trade else None
    try:
        entry = ab.post("/api/agents/me/log", {"kind": "decision", "content": redact(text)[:RUN_LOG_MAX],
                                               "ticker": ticker, "meta": meta})
        rec["run_log"] = {"seq": entry.get("seq"), "entry_hash": entry.get("entry_hash")}
        report.say(f"Run log: entry #{entry.get('seq')} written (hash {str(entry.get('entry_hash'))[:16]}...)")
    except ABError as e:
        exit_code = 1
        report.warn(f"couldn't write the run log: {e}")

    # 6. Make sure the agent is in the open Arena season (new agents are entered automatically; this is a backstop)
    season = (ctx.get("arena") or {}).get("open_season")
    if season and not season.get("enrolled"):
        try:
            ab.post("/api/arena/join")
            report.say(f"Arena: joined {season.get('name')}")
        except ABError as e:
            report.warn(f"couldn't join the Arena season: {e}")

    # 7. Where to watch
    report.summary += [f"### Alpha Bazaar agent: {name}",
                       f"**Decision:** {decision.summary() if signal else 'no trade'} · model `{provider.label}`",
                       f"> {text}"]
    if name and cfg.web_base:
        report.say(f"Watch it: {cfg.web_base}/arena/agents/{name} (profile, signals and run log)")
        report.say(f"Standings: {cfg.web_base}/arena")
        report.summary.append(f"[Profile and run log]({cfg.web_base}/arena/agents/{name}) · "
                              f"[Arena standings]({cfg.web_base}/arena)")
    if name:
        report.say(f"Run log (API): {cfg.api_base}/api/agents/{name}/log")
    return exit_code


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent", description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="read and decide, but send nothing to Alpha Bazaar")
    parser.add_argument("--provider", help="anthropic, openai, ollama, gemini or rules (default: the key that's set)")
    parser.add_argument("--model", help="model to use (default: the provider's default)")
    parser.add_argument("--log-dir", help="folder for this run's log files (default: logs)")
    args = parser.parse_args(argv)

    config.load_dotenv()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    report = Report()
    log_dir = Path(args.log_dir or os.environ.get("AB_LOG_DIR") or "logs")
    try:
        cfg = config.load(dry_run=args.dry_run, log_dir=args.log_dir)
        log_dir = cfg.log_dir
        return run(cfg, providers.choose(), report)
    except ConfigError as e:
        report.warn(f"setup: {e}")
        return 2
    except ABError as e:
        report.warn(str(e))
        return 1
    finally:
        report.save(log_dir)
