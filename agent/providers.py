"""Model providers. The first key that's set picks one (or set LLM_PROVIDER to choose):

  ANTHROPIC_API_KEY -> Anthropic (Claude), default model claude-opus-5-5
  OPENAI_BASE_URL   -> Ollama or any OpenAI-compatible server (set LLM_MODEL too)
  OPENAI_API_KEY    -> OpenAI, default model gpt-6.1-sol
  GEMINI_API_KEY    -> Google Gemini (works on the free tier), default model gemini-3.8-flash
  none of them      -> the deterministic rules in rules.py: free, no model at all

Each provider returns the model's raw reply. decision.validate() checks it the same way, whichever model wrote it.
"""
import re
import time
from typing import Any, Dict, Optional

from agent import rules
from agent.config import ConfigError, env
from agent.decision import SCHEMA

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-6.1-sol", "gemini": "gemini-3.8-flash"}
OPENAI_URL = "https://api.openai.com/v1"
OLLAMA_URL = "http://localhost:11434/v1"
TIMEOUT = 180.0  # seconds for one model call
RETRIES = 2  # the SDKs' own retries (rate limits, overload, timeouts): the whole run must fit the job's 15 minutes
MAX_TOKENS = 16000  # room for the model's thinking plus a short JSON answer
RULES_LABEL = "rules-based (no LLM)"
# Claude models that take server-side refusal fallbacks (fallbacks="default" picks the backup model by itself)
CLAUDE_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")
_LABEL_CHARS = re.compile(r"[^A-Za-z0-9 ._:/@+()\-]")


class ProviderError(Exception):
    """The model call failed. fatal=True means waiting won't fix it (a bad key or model name): check the settings."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


def _require(module: str, package: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        raise ProviderError(f"the {package} package isn't installed: pip install -r requirements.txt",
                            fatal=True) from None


class Provider:
    name = "provider"

    def __init__(self, model: str):
        self.model = model

    @property
    def label(self) -> str:
        """The model as your public Arena profile shows it (and groups the by-model standings)."""
        label = re.sub(r"^[^A-Za-z0-9]+", "", _LABEL_CHARS.sub("-", self.model)).rstrip(" -")
        return label[:100] or "unknown"  # AB: up to 100 characters, starting with a letter or digit

    def decide(self, system: str, user: str, context: Dict[str, Any], candidates: Dict[str, Any]) -> Any:
        raise NotImplementedError


class Anthropic(Provider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str, effort: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key
        # Claude Opus 5.5 thinks before it answers; effort sets how much (its default is medium, set here on purpose)
        self.effort = effort or ("medium" if model == DEFAULT_MODELS["anthropic"] else None)
        self.base_url = base_url

    def decide(self, system, user, context=None, candidates=None):
        anthropic = _require("anthropic", "anthropic")
        options = {"base_url": self.base_url} if self.base_url else {}
        client = anthropic.Anthropic(api_key=self.api_key, max_retries=RETRIES, timeout=TIMEOUT, **options)
        base = {"model": self.model, "max_tokens": MAX_TOKENS, "system": system,
                "messages": [{"role": "user", "content": user}]}
        output_config: Dict[str, Any] = {"format": {"type": "json_schema", "schema": SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort
        extras: Dict[str, Any] = {}
        if self.model.startswith(CLAUDE_FALLBACK_MODELS):
            # If a safety classifier declines, the API re-runs the request on a fallback model it picks
            extras = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            try:
                response = client.beta.messages.create(**base, output_config=output_config, **extras)
            except anthropic.BadRequestError:
                # An older model may not take structured output, effort or fallbacks: ask once more without them
                response = client.messages.create(**base)
        except anthropic.AuthenticationError:
            raise ProviderError("Anthropic rejected ANTHROPIC_API_KEY (401)", fatal=True) from None
        except anthropic.PermissionDeniedError as e:
            raise ProviderError(f"Anthropic refused access (403): {e.message}", fatal=True) from None
        except anthropic.NotFoundError:
            raise ProviderError(f"Anthropic has no model {self.model!r}: fix LLM_MODEL", fatal=True) from None
        except anthropic.BadRequestError as e:
            raise ProviderError(f"Anthropic rejected the request (400): {e.message}", fatal=True) from None
        except anthropic.RateLimitError:
            raise ProviderError("Anthropic is rate limiting this key (429); it will try again tomorrow") from None
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Anthropic API error {e.status_code}") from None
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"couldn't reach Anthropic ({type(e).__name__})") from None

        if response.stop_reason == "refusal":
            raise ProviderError("Claude declined to answer (stop_reason refusal)")
        if response.stop_reason == "max_tokens":
            raise ProviderError("Claude's answer was cut off (max_tokens)")
        return next((block.text for block in response.content if block.type == "text"), "")


class OpenAI(Provider):
    """OpenAI, or any server that speaks its Chat Completions API: Ollama, LM Studio, vLLM, Groq and more."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 effort: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key
        self.compatible = bool(base_url) and base_url.rstrip("/") != OPENAI_URL
        self.base_url = base_url.rstrip("/") if base_url else OPENAI_URL
        self.effort = effort
        self.name = "openai-compatible" if self.compatible else "openai"

    def decide(self, system, user, context=None, candidates=None):
        openai = _require("openai", "openai")
        # Ollama and most local servers ignore the key, but the client needs one
        client = openai.OpenAI(api_key=self.api_key or "not-needed", base_url=self.base_url, max_retries=RETRIES,
                               timeout=TIMEOUT)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.compatible:
            formats = [{"type": "json_object"}, None]
        else:
            formats = [{"type": "json_schema", "json_schema": {"name": "trade_decision", "strict": True,
                                                                 "schema": SCHEMA}}, None]
        where = "OpenAI" if not self.compatible else self.base_url
        for response_format in formats:
            request: Dict[str, Any] = {"model": self.model, "messages": messages}
            if response_format:
                request["response_format"] = response_format
            if self.effort and not self.compatible:
                request["reasoning_effort"] = self.effort
            try:
                completion = client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                if response_format is not None:
                    continue  # this model or server doesn't do that JSON mode: ask in plain text instead
                raise ProviderError(f"{where} rejected the request (400): {e.message}", fatal=True) from None
            except openai.AuthenticationError:
                raise ProviderError(f"{where} rejected the API key (401): check OPENAI_API_KEY", fatal=True) from None
            except openai.PermissionDeniedError as e:
                raise ProviderError(f"{where} refused access (403): {e.message}", fatal=True) from None
            except openai.NotFoundError:
                raise ProviderError(f"{where} has no model {self.model!r}: fix LLM_MODEL", fatal=True) from None
            except openai.RateLimitError as e:
                if "insufficient_quota" in str(e):
                    raise ProviderError(f"{where} says the account is out of credits (429)", fatal=True) from None
                raise ProviderError(f"{where} is rate limiting this key (429)") from None
            except openai.APIStatusError as e:
                raise ProviderError(f"{where} API error {e.status_code}") from None
            except openai.APIConnectionError as e:
                hint = " Is the server running and reachable from here?" if self.compatible else ""
                raise ProviderError(f"couldn't reach {where} ({type(e).__name__}).{hint}") from None

            choice = completion.choices[0] if completion.choices else None
            if choice is None:
                raise ProviderError(f"{where} returned no answer")
            refusal = getattr(choice.message, "refusal", None)
            if refusal:
                raise ProviderError(f"the model declined: {refusal[:200]}")
            if choice.finish_reason == "length":
                raise ProviderError("the model's answer was cut off (finish_reason length)")
            return choice.message.content or ""
        raise ProviderError(f"{where} rejected every request format")  # not reached: the last format raises


class Gemini(Provider):
    name = "gemini"
    BUSY_WAITS = (5, 20)  # the free tier answers 503 "high demand" now and then: wait and try again, twice

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key
        self.base_url = base_url

    def decide(self, system, user, context=None, candidates=None):
        genai = _require("google.genai", "google-genai")
        from google.genai import errors, types
        import httpx

        http_options = types.HttpOptions(timeout=int(TIMEOUT * 1000), base_url=self.base_url)
        client = genai.Client(api_key=self.api_key, http_options=http_options)
        config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json")
        for wait in (*self.BUSY_WAITS, None):
            try:
                response = client.models.generate_content(model=self.model, contents=user, config=config)
                break
            except errors.APIError as e:
                code = e.code
                if code in (429, 500, 502, 503, 504) and wait is not None:
                    time.sleep(wait)
                    continue
                detail = (e.message or e.status or "")[:200]
                if code == 404:
                    raise ProviderError(f"Gemini has no model {self.model!r} for this key: fix LLM_MODEL",
                                        fatal=True) from None
                if code in (400, 401, 403):
                    raise ProviderError(f"Gemini rejected the request ({code}): {detail}", fatal=True) from None
                raise ProviderError(f"Gemini API error {code}: {detail}") from None
            except httpx.HTTPError as e:
                raise ProviderError(f"couldn't reach Gemini ({type(e).__name__})") from None

        text = response.text
        if not text:
            finish = response.candidates[0].finish_reason if response.candidates else None
            raise ProviderError(f"Gemini returned no text (finish reason: {finish})")
        return text


class Rules(Provider):
    """No model: the deterministic momentum rule in rules.py. Free, and the same input gives the same answer."""

    name = "rules"

    def __init__(self):
        super().__init__("rules")

    @property
    def label(self) -> str:
        return RULES_LABEL

    def decide(self, system, user, context, candidates):
        return rules.decide(context, candidates)


_ALIASES = {"claude": "anthropic", "google": "gemini", "ollama": "openai-compatible", "compatible": "openai-compatible",
            "local": "openai-compatible", "none": "rules", "no-llm": "rules"}


def choose() -> Provider:
    """The provider for this run: LLM_PROVIDER if set, else the first model key that's set, else the rules."""
    forced = (env("LLM_PROVIDER") or "").lower()
    forced = _ALIASES.get(forced, forced)
    model = env("LLM_MODEL")
    anthropic_key = env("ANTHROPIC_API_KEY")
    openai_key = env("OPENAI_API_KEY")
    gemini_key = env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    base_url = env("OPENAI_BASE_URL")
    effort = env("LLM_EFFORT")

    name = forced or ("anthropic" if anthropic_key else "openai-compatible" if base_url
                      else "openai" if openai_key else "gemini" if gemini_key else "rules")
    if name == "anthropic":
        if not anthropic_key:
            raise ConfigError("LLM_PROVIDER is anthropic, but ANTHROPIC_API_KEY isn't set")
        return Anthropic(model or DEFAULT_MODELS["anthropic"], anthropic_key, effort=effort)
    if name == "openai":
        if not openai_key:
            raise ConfigError("LLM_PROVIDER is openai, but OPENAI_API_KEY isn't set")
        return OpenAI(model or DEFAULT_MODELS["openai"], openai_key, base_url=OPENAI_URL, effort=effort)
    if name == "openai-compatible":
        if not model:
            raise ConfigError("set LLM_MODEL to a model your server serves (for Ollama, one from `ollama list`, "
                              "such as llama3.2)")
        return OpenAI(model, openai_key, base_url=base_url or OLLAMA_URL)
    if name == "gemini":
        if not gemini_key:
            raise ConfigError("LLM_PROVIDER is gemini, but GEMINI_API_KEY isn't set")
        return Gemini(model or DEFAULT_MODELS["gemini"], gemini_key)
    if name == "rules":
        return Rules()
    raise ConfigError(f"LLM_PROVIDER must be anthropic, openai, ollama, gemini or rules, not {forced!r}")
