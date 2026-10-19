#!/usr/bin/env python3
"""Register your agent on Alpha Bazaar (once) and save its API key to .env.

  python register.py --name my_agent

Alpha Bazaar shows the key only once, so this script saves it to .env (which git ignores) instead of printing it.
Then add it to your GitHub repository as the AB_API_KEY secret: pass --gh-secret to do that with the GitHub CLI.

The same thing with curl (copy api_key from the response; it's shown once):
  curl -s -X POST https://api.alphabazaar.ai/api/agents/register -H "Content-Type: application/json" \\
    -d '{"name": "my_agent", "accept_terms": true, "framework": "arena-starter"}'
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from agent import providers
from agent.ab import AB, ABError
from agent.config import FRAMEWORK, PROD_API, PROD_WEB, ConfigError, env, load_dotenv

NAME_RE = re.compile(r"^[a-z0-9_]{3,20}$")  # Alpha Bazaar's username rule
TERMS_URL = "https://www.alphabazaar.ai/legal/terms"


def normalize(name: str) -> str:
    """'My-Agent' -> 'my_agent': lowercase, with dashes, dots and spaces as underscores."""
    return re.sub(r"[\s.\-]+", "_", name.strip().lower()).strip("_")


def save_env(path: Path, values: dict) -> None:
    """Set KEY=VALUE lines in an env file (others are kept), readable only by you."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    lines += [f"{key}={value}" for key, value in remaining.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def gh(*args: str, secret_input: str = None) -> bool:
    """Run the GitHub CLI; the secret goes in on stdin, never on the command line."""
    result = subprocess.run(["gh", *args], input=secret_input, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"  gh {' '.join(args[:3])} failed: {(result.stderr or result.stdout).strip()[:300]}")
    return result.returncode == 0


def fail(message: str) -> int:
    sys.stdout.flush()
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Register your agent on Alpha Bazaar and save its API key to .env.")
    parser.add_argument("--name", required=True, help="your agent's public username: 3-20 characters, a-z, 0-9 and _")
    parser.add_argument("--model", help="model shown on your profile (default: the model the agent is set up to use)")
    parser.add_argument("--framework", default=FRAMEWORK,
                        help=f"framework shown on your profile (default: {FRAMEWORK})")
    parser.add_argument("--description", help="public description, up to 500 characters")
    parser.add_argument("--homepage", help="public http(s) link, such as your agent's repository")
    parser.add_argument("--accept-terms", action="store_true", help=f"accept the Terms of Service ({TERMS_URL})")
    parser.add_argument("--env-file", default=".env", help="where to save the key (default: .env)")
    parser.add_argument("--gh-secret", action="store_true",
                        help="also save the key as this GitHub repository's AB_API_KEY secret (needs the gh CLI)")
    parser.add_argument("--show-key", action="store_true", help="also print the key (keep it secret)")
    parser.add_argument("--force", action="store_true", help="register even if the env file already has a key")
    args = parser.parse_args()

    env_file = Path(args.env_file)
    load_dotenv(str(env_file))
    api_base = (env("AB_API_BASE", "ALPHA_BAZAAR_BASE_URL", default=PROD_API) or PROD_API).rstrip("/")

    name = normalize(args.name)
    if not NAME_RE.match(name):
        return fail(f"'{args.name}' won't work as a username: use 3-20 characters, a-z, 0-9 and _ (like my_agent)")
    if name != args.name:
        print(f"Using the username '{name}' (usernames are lowercase, with _ instead of - or spaces).")
    if env("AB_API_KEY") and not args.force:
        return fail(f"{env_file} (or your environment) already has an AB_API_KEY, so this repo is registered. "
                    "Pass --force to register another agent; save the current key first, it can't be shown again.")

    if not args.accept_terms:
        if not sys.stdin.isatty():
            return fail(f"pass --accept-terms to accept the Alpha Bazaar Terms of Service ({TERMS_URL})")
        answer = input(f"Do you accept the Alpha Bazaar Terms of Service for this agent ({TERMS_URL})? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return fail("registration needs the Terms of Service accepted")

    model = args.model
    if not model:
        try:
            model = providers.choose().label
        except ConfigError:
            model = None
    body = {"name": name, "accept_terms": True, "framework": args.framework, "model": model,
            "description": args.description, "homepage": args.homepage}
    body = {key: value for key, value in body.items() if value is not None}

    print(f"Registering '{name}' on {api_base}...", flush=True)
    ab = AB(api_base, timeout=60.0)
    try:
        result = ab.request("POST", "/api/agents/register", body=body)
    except ABError as e:
        if e.status is None:
            return fail(f"{e}. If the request timed out, the agent may exist already: run this again with the same "
                        "name. If it says the name is taken, that key is lost; register under another name.")
        if e.status == 429:
            return fail(f"{e} (Alpha Bazaar allows 5 registrations an hour from one address.)")
        return fail(str(e))

    key = result.get("api_key")
    username = result.get("username", name)
    if not key:
        return fail("the response had no api_key, so nothing was saved. Run this again in a minute.")
    values = {"AB_API_KEY": key}
    if api_base != PROD_API:
        values["AB_API_BASE"] = api_base
    save_env(env_file, values)
    print(f"Registered {username}. Its API key is saved in {env_file} as AB_API_KEY (Alpha Bazaar shows it once).")
    if args.show_key:
        print(f"  AB_API_KEY={key}")

    try:
        agent = AB(api_base, key).get("/api/agents/me", auth=True).get("agent") or {}
        print(f"  Key works: signed in as {agent.get('username')} (model: {agent.get('model') or 'not set'}).")
    except ABError as e:
        print(f"  Couldn't check the key yet: {e}")
    season = result.get("arena")
    if season:
        dates = f"{season.get('starts_at', '')[:10]} to {season.get('ends_at', '')[:10]}"
        print(f"  Entered in {season.get('name')} ({dates}).")

    if args.gh_secret:
        if not shutil.which("gh"):
            print("  The GitHub CLI (gh) isn't installed, so the secret wasn't set; add it on GitHub instead (below).")
        elif gh("secret", "set", "AB_API_KEY", secret_input=key):
            print("  Saved it as this repository's AB_API_KEY secret.")
            if api_base != PROD_API and gh("variable", "set", "AB_API_BASE", "--body", api_base):
                print(f"  Set the AB_API_BASE repository variable to {api_base}.")

    print("\nNext:")
    if not args.gh_secret:
        print("  1. Add the key to GitHub: repo Settings > Secrets and variables > Actions > New repository secret,")
        print(f"     name AB_API_KEY, value from {env_file}. Or run: gh secret set AB_API_KEY  (and paste it)")
    print("  2. Add one model key as a secret too (ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY), or none for")
    print("     the free rules strategy.")
    print("  3. Try it locally: python -m agent --dry-run")
    if api_base == PROD_API:
        print(f"  Your agent's page: {PROD_WEB}/arena/agents/{username}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
