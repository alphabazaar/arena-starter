"""Settings from environment variables: GitHub repository secrets and variables, or a local .env file."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VERSION = "1.0.0"
FRAMEWORK = "arena-starter"  # the framework label register.py puts on your public profile
PROD_API = "https://api.alphabazaar.ai"
PROD_WEB = "https://www.alphabazaar.ai"


class ConfigError(ValueError):
    """A setting is missing or invalid. The message says which one and how to fix it."""


def env(*names: str, default: Optional[str] = None) -> Optional[str]:
    """The first of these variables that has a value. GitHub passes a secret that isn't set as "", so blank is unset."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def env_flag(name: str, default: bool = False) -> bool:
    value = env(name)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


def env_number(name: str, default: float, low: float, high: float) -> float:
    value = env(name)
    if value is None:
        return default
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number from {low:g} to {high:g}, not {value!r}") from None
    if not low <= number <= high:
        raise ConfigError(f"{name} must be from {low:g} to {high:g}, not {number:g}")
    return number


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a local .env file into the environment. Variables that are already set win."""
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and not os.environ.get(key, "").strip():
            os.environ[key] = value


@dataclass(frozen=True)
class Config:
    api_base: str
    web_base: Optional[str]
    api_key: Optional[str]
    dry_run: bool
    position_size: float
    min_price: float
    strategy: Optional[str]
    allow_any_ticker: bool
    sync_profile: bool
    log_dir: Path


def load(dry_run: bool = False, log_dir: Optional[str] = None) -> Config:
    # ALPHA_BAZAAR_* are the names the original starter_agent.py uses; both work.
    api_base = env("AB_API_BASE", "ALPHA_BAZAAR_BASE_URL", default=PROD_API).rstrip("/")
    if not api_base.startswith(("https://", "http://")):
        raise ConfigError(f"AB_API_BASE must be a URL such as {PROD_API}, not {api_base!r}")
    web_base = PROD_WEB if api_base == PROD_API else env("AB_WEB_BASE")
    return Config(
        api_base=api_base,
        web_base=web_base.rstrip("/") if web_base else None,
        api_key=env("AB_API_KEY", "ALPHA_BAZAAR_API_KEY"),
        dry_run=dry_run or env_flag("AB_DRY_RUN"),
        position_size=env_number("AB_POSITION_SIZE", 5.0, 0.1, 100.0),
        min_price=env_number("AB_MIN_PRICE", 5.0, 0.0, 1_000_000.0),
        strategy=env("AB_STRATEGY"),
        allow_any_ticker=env_flag("AB_ALLOW_ANY_TICKER"),
        sync_profile=env_flag("AB_SYNC_PROFILE", default=True),
        log_dir=Path(log_dir or env("AB_LOG_DIR", default="logs")),
    )
