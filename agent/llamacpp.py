"""Run an open-weight model on the GitHub runner itself: a pinned llama.cpp release and a GGUF from Hugging Face, both
sha256-checked, served by llama-server on 127.0.0.1. The agent then talks to it like any OpenAI-compatible server.

The workflow (trade.yml) runs these steps when LLM_PROVIDER is llamacpp:

  python -m agent.llamacpp plan  --dirs /mnt/llamacpp "$RUNNER_TEMP/llamacpp"   # settings, size guard, cache keys
  python -m agent.llamacpp fetch                                                # download what the cache lacked, check sha256
  python -m agent.llamacpp start                                                # llama-server, wait until healthy
  python -m agent                                                               # the usual agent, pointed at it
  python -m agent.llamacpp stop

Settings (repository variables; all optional: with none set, the League's Qwen3.5 4B runs):
  LLAMA_GGUF_REPO, LLAMA_GGUF_FILE     a public, ungated Hugging Face repository and one .gguf file in it
  LLAMA_GGUF_REVISION                  a commit (best) or a branch/tag, default main; the run log names the commit
  LLAMA_GGUF_SHA256                    the file's sha256; without it the run prints the sha256 it computed
  LLAMA_CTX_SIZE, LLAMA_MAX_TOKENS     context window and answer budget (thinking included)
  LLAMA_TIMEOUT_MINUTES                how long one answer may take
  LLAMA_SERVER_ARGS                    extra llama-server flags, such as --reasoning-budget 0
  LLAMA_MAX_GGUF_GB                    the size guard, default 12.5 (a GitHub runner has 16 GB of RAM)

The download, verification and llama-server steps follow Alpha Bazaar's ai-league-runner
(https://github.com/alphabazaar/ai-league-runner, Apache-2.0): see NOTICE.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.config import ConfigError, env, env_number

USER_AGENT = "AlphaBazaar-ArenaStarter/1.0 (+https://github.com/alphabazaar/arena-starter)"
HF = "https://huggingface.co"
GB = 1000 ** 3  # sizes as Hugging Face shows them

# The pinned llama.cpp CPU build, the same one the AI League runs (ai-league-runner's models.json)
LLAMA_CPP = {
    "tag": "b12387",
    "commit": "a956a852c9c8bc1fb07206364838ddef299178c0",
    "url": "https://github.com/ggml-org/llama.cpp/releases/download/b12387/llama-b12387-bin-ubuntu-x64.tar.gz",
    "sha256": "9c58ebcd5a704e879cfadf2bbcb1fb1ec8ad4a5bec761fc58b3515864dd5e37a",
}
# The default model: the League's Qwen3.5 4B, pinned exactly as ai-league-runner pins it. 3.0 GB, answers on 4 CPUs.
DEFAULT_GGUF = {
    "repo": "ggml-org/Qwen3.5-4B-GGUF",
    "file": "Qwen3.5-4B-Q4_K_M.gguf",
    "revision": "8d43d974efc48399cb58957ad49c8b26bf5a109c",
    "sha256": "217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14",
}
DEFAULT_CTX = 24576  # today's prompt is about 7k tokens; the rest is room for the answer
DEFAULT_MAX_TOKENS = 8192  # thinking included: Qwen3.5 4B wrote 4,438 tokens on a real day's prompt
DEFAULT_TIMEOUT_MINUTES = 55  # measured on a GitHub runner: ~3.5 tokens/s with a 7k-token prompt, so 8192 tokens ~40 min
DEFAULT_MAX_GB = 12.5  # the file must fit in a 16 GB runner's RAM next to its context (gpt-oss-20b, 12.1 GB, just does)
PORT = 8080
SERVER_START_SECONDS = 10 * 60
DISK_SPARE = 2 * GB
STATE_FILE = "plan.json"
PROVENANCE_FILE = "provenance.json"
SERVER_LOG = "llama-server.log"
PID_FILE = "llama-server.pid"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FILE = re.compile(r"^(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.gguf$")
_SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
_REVISION = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_SERVER_ARG = re.compile(r"^[-A-Za-z0-9_.=:,]+$")
# Flags the starter sets itself: the model file (only the checked one runs), where it listens, and its name
_RESERVED_ARGS = {"-m", "--model", "-mu", "--model-url", "-hf", "-hfr", "--hf-repo", "-hff", "--hf-file", "--host",
                  "--port", "-a", "--alias", "--api-key", "--api-key-file", "-c", "--ctx-size"}


class LlamaError(ConfigError):
    """Something this mode can't do (a bad setting, a file too big, a wrong sha256). Its message is safe to print."""


# --- Settings -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    repo: str
    file: str
    revision: str
    sha256: Optional[str]  # what the operator pinned; None means "trust Hugging Face's and print what we computed"
    ctx: int
    max_tokens: int
    timeout: float  # seconds for one answer
    port: int
    server_args: Tuple[str, ...]
    max_bytes: int
    is_default: bool

    @property
    def alias(self) -> str:
        """The model's name: the file's, without .gguf (Qwen3.5-4B-Q4_K_M). Also the profile's model label."""
        return Path(self.file).name[: -len(".gguf")]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def settings() -> Settings:
    """The llama.cpp settings from the environment, checked. With no LLAMA_GGUF_* set, the pinned default model."""
    repo, file = env("LLAMA_GGUF_REPO"), env("LLAMA_GGUF_FILE")
    revision, sha256 = env("LLAMA_GGUF_REVISION"), env("LLAMA_GGUF_SHA256")
    is_default = not any((repo, file, revision, sha256))
    if is_default:
        repo, file, revision, sha256 = (DEFAULT_GGUF[k] for k in ("repo", "file", "revision", "sha256"))
    elif not repo or not file:
        raise LlamaError("set both LLAMA_GGUF_REPO (such as ggml-org/Qwen3.5-4B-GGUF) and LLAMA_GGUF_FILE (such as "
                         "Qwen3.5-4B-Q4_K_M.gguf), or neither to run the default model")
    repo = repo.strip().strip("/")
    if repo.startswith(HF + "/"):
        repo = repo[len(HF) + 1:]
    if not _REPO.match(repo):
        raise LlamaError(f"LLAMA_GGUF_REPO must be a Hugging Face repository such as ggml-org/Qwen3.5-4B-GGUF, "
                         f"not {repo!r}")
    if not _FILE.match(file) or ".." in file.split("/"):
        raise LlamaError(f"LLAMA_GGUF_FILE must be a .gguf file name such as Qwen3.5-4B-Q4_K_M.gguf, not {file!r}")
    if _SPLIT.search(file):
        raise LlamaError(f"{file} is one part of a split GGUF; pick a single-file quant (split files aren't "
                         f"supported here)")
    revision = revision or "main"
    if not _REVISION.match(revision):
        raise LlamaError(f"LLAMA_GGUF_REVISION must be a commit (40 hex characters) or a branch or tag name, "
                         f"not {revision!r}")
    if sha256 is not None:
        sha256 = sha256.lower()
        if not _SHA256.match(sha256):
            raise LlamaError("LLAMA_GGUF_SHA256 must be 64 hex characters (the sha256 Hugging Face shows for the file)")
    raw_args = env("LLAMA_SERVER_ARGS") or ""
    server_args = tuple(raw_args.split())
    for arg in server_args:
        if not _SERVER_ARG.match(arg):
            raise LlamaError(f"LLAMA_SERVER_ARGS may only hold flags and plain values, not {arg!r}")
        if arg.split("=", 1)[0] in _RESERVED_ARGS:
            raise LlamaError(f"LLAMA_SERVER_ARGS can't set {arg.split('=', 1)[0]}: the starter sets it (use "
                             f"LLAMA_GGUF_* for the model and LLAMA_CTX_SIZE for the context)")
    return Settings(
        repo=repo, file=file, revision=revision, sha256=sha256,
        ctx=int(env_number("LLAMA_CTX_SIZE", DEFAULT_CTX, 2048, 262144)),
        max_tokens=int(env_number("LLAMA_MAX_TOKENS", DEFAULT_MAX_TOKENS, 256, 65536)),
        timeout=env_number("LLAMA_TIMEOUT_MINUTES", DEFAULT_TIMEOUT_MINUTES, 1, 300) * 60,
        port=int(env_number("LLAMA_PORT", PORT, 1024, 65535)),
        server_args=server_args,
        max_bytes=int(env_number("LLAMA_MAX_GGUF_GB", DEFAULT_MAX_GB, 0.1, 1024) * GB),
        is_default=is_default,
    )


def gguf_url(repo: str, revision: str, file: str) -> str:
    return f"{HF}/{repo}/resolve/{revision}/{file}"


def size_words(size: int) -> str:
    return f"{size / GB:.1f} GB"


def check_size(size: Optional[int], limit: int, file: str) -> None:
    """The size guard: a GGUF bigger than the limit won't fit in the runner's memory, so it isn't even downloaded."""
    if size is not None and size > limit:
        raise LlamaError(
            f"{file} is {size_words(size)}, more than the {size_words(limit)} that fits next to its context in a "
            f"GitHub runner's 16 GB of RAM. Pick a smaller quant (an 8B model's Q4_K_M is about 5 GB), or raise "
            f"LLAMA_MAX_GGUF_GB if you run on a bigger self-hosted runner.")


# --- Hugging Face -----------------------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # keep the first answer: its headers describe the file
        return None


def http_head(url: str, timeout: float = 30) -> Tuple[int, Dict[str, str]]:
    """(status, lower-cased headers) of a HEAD request, without following redirects. Anonymous: no token is sent."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as e:
        e.close()
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}
    except (urllib.error.URLError, OSError) as e:
        raise LlamaError(f"couldn't reach Hugging Face ({type(e).__name__})") from None


@dataclass
class Remote:
    """What Hugging Face says about the file: the commit the revision points at, its size and its sha256."""
    commit: str
    size: Optional[int]
    sha256: Optional[str]


def describe_remote(s: Settings, head: Callable[[str], Tuple[int, Mapping[str, str]]] = http_head) -> Remote:
    status, headers = head(gguf_url(s.repo, s.revision, s.file))
    if status in (401, 403):
        raise LlamaError(f"{s.repo} is private or gated on Hugging Face ({status}). This mode downloads anonymously: "
                         f"pick a public, ungated GGUF repository")
    if status == 404:
        raise LlamaError(f"Hugging Face has no {s.file} in {s.repo} at {s.revision} (404): check LLAMA_GGUF_REPO, "
                         f"LLAMA_GGUF_FILE and LLAMA_GGUF_REVISION")
    if status >= 400:
        raise LlamaError(f"Hugging Face answered {status} for {s.repo}/{s.file}")
    commit = (headers.get("x-repo-commit") or "").strip().lower()
    if not _COMMIT.match(commit):
        commit = s.revision if _COMMIT.match(s.revision) else ""
    if not commit:
        raise LlamaError(f"Hugging Face didn't say which commit {s.revision} is; set LLAMA_GGUF_REVISION to a commit")
    size_text = headers.get("x-linked-size") or (headers.get("content-length") if status == 200 else None)
    size = int(size_text) if size_text and size_text.isdigit() else None
    etag = (headers.get("x-linked-etag") or "").strip().strip('"').removeprefix("W/").strip('"').lower()
    return Remote(commit=commit, size=size, sha256=etag if _SHA256.match(etag) else None)


# --- Files ------------------------------------------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: Optional[str], what: str) -> str:
    """The file's sha256, which must be `expected` when there is one. A mismatch deletes the file."""
    actual = sha256_file(path)
    if expected and actual != expected:
        path.unlink(missing_ok=True)
        raise LlamaError(f"{what}: sha256 mismatch. Expected {expected}, got {actual}. The file was deleted; nothing "
                         f"ran. If you changed the model, update LLAMA_GGUF_SHA256")
    return actual


def download(url: str, dest: Path, max_bytes: int, *, retries: int = 3, timeout: float = 60,
             opener: Callable[..., Any] = urllib.request.urlopen, sleep: Callable[[float], None] = time.sleep) -> int:
    """Download url to dest (through dest.part, resuming where a dropped connection left off). Anonymous. Stops as soon
    as the file passes max_bytes, so the size guard holds even when Hugging Face didn't say how big the file is."""
    part = dest.with_name(dest.name + ".part")
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with opener(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status not in (200, 206):
                    raise LlamaError(f"download answered {status}")
                mode = "ab" if status == 206 and have else "wb"
                written = have if mode == "ab" else 0
                with open(part, mode) as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > max_bytes:
                            handle.close()
                            part.unlink(missing_ok=True)
                            raise LlamaError(f"{dest.name} is bigger than the {size_words(max_bytes)} limit "
                                             f"(LLAMA_MAX_GGUF_GB); the download was stopped")
                        handle.write(block)
            part.replace(dest)
            return written
        except LlamaError:
            raise
        except urllib.error.HTTPError as e:
            e.close()
            if e.code == 416 and have:  # the part is already complete (or bad): start over
                part.unlink(missing_ok=True)
            elif e.code < 500 and e.code != 429:
                raise LlamaError(f"download of {dest.name} answered {e.code}") from None
            last = e
        except (urllib.error.URLError, OSError) as e:
            last = e
        if attempt < retries:
            sleep(5 * (attempt + 1))
    raise LlamaError(f"couldn't download {dest.name} ({type(last).__name__ if last else 'unknown error'})")


def extract_llama(archive: Path, dest: Path) -> Path:
    """Unpack the llama.cpp release into dest; returns the folder holding llama-server (and its libraries)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    found = sorted(dest.rglob("llama-server"), key=lambda p: len(p.parts))
    if not found or not found[0].is_file():
        raise LlamaError("the llama.cpp archive has no llama-server in it")
    found[0].chmod(found[0].stat().st_mode | 0o111)
    return found[0].parent


def pick_dir(candidates: Sequence[str], need: int,
             free: Callable[[str], int] = lambda d: shutil.disk_usage(d).free) -> Path:
    """The first folder that can be created and has room for the files."""
    best = ""
    for candidate in candidates:
        try:
            Path(candidate).mkdir(parents=True, exist_ok=True)
            available = free(candidate)
        except OSError:
            continue
        if available >= need:
            return Path(candidate)
        best = best or f"{candidate} has {size_words(available)} free"
    raise LlamaError(f"no disk has room for {size_words(need)}" + (f" ({best})" if best else ""))


# --- The plan: what this run downloads and runs -----------------------------------------------------------------------

def run_url(environ: Mapping[str, str] = os.environ) -> Optional[str]:
    run_id = environ.get("GITHUB_RUN_ID")
    if not run_id or not environ.get("GITHUB_REPOSITORY"):
        return None
    return f"{environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}"


@dataclass
class Plan:
    dir: str
    repo: str
    file: str
    revision: str  # as set
    commit: str  # what it pointed at: the download uses this, so the file can't change mid-run
    size: Optional[int]
    expected_sha256: Optional[str]  # the operator's pin, else Hugging Face's
    pinned: bool  # the operator (or the default) set the sha256
    alias: str
    ctx: int
    port: int
    server_args: List[str]
    max_bytes: int
    llama_cpp: Dict[str, str] = field(default_factory=lambda: dict(LLAMA_CPP))

    @property
    def gguf_path(self) -> Path:
        return Path(self.dir) / "models" / Path(self.file).name

    @property
    def llama_archive(self) -> Path:
        return Path(self.dir) / f"llama-{self.llama_cpp['tag']}.tar.gz"

    @property
    def gguf_cache_key(self) -> str:
        if self.expected_sha256:
            return f"gguf-{self.expected_sha256}"
        return "gguf-" + re.sub(r"[^A-Za-z0-9._-]", "-", f"{self.repo}-{self.commit}-{self.file}")

    @property
    def llama_cache_key(self) -> str:
        return f"llamacpp-{self.llama_cpp['tag']}-{self.llama_cpp['sha256'][:16]}"


def make_plan(s: Settings, dirs: Sequence[str], head=http_head, free=None) -> Plan:
    remote = describe_remote(s, head)
    if s.sha256 and remote.sha256 and s.sha256 != remote.sha256:
        raise LlamaError(f"LLAMA_GGUF_SHA256 is {s.sha256}, but Hugging Face says {s.file} at {remote.commit[:12]} is "
                         f"{remote.sha256}. Fix the sha256 or the revision; nothing was downloaded")
    check_size(remote.size, s.max_bytes, s.file)
    need = (remote.size or s.max_bytes) + DISK_SPARE
    directory = pick_dir(dirs, need, free) if free else pick_dir(dirs, need)
    return Plan(dir=str(directory), repo=s.repo, file=s.file, revision=s.revision, commit=remote.commit,
                size=remote.size, expected_sha256=s.sha256 or remote.sha256, pinned=bool(s.sha256), alias=s.alias,
                ctx=s.ctx, port=s.port, server_args=list(s.server_args), max_bytes=s.max_bytes)


def save_plan(plan: Plan) -> Path:
    path = Path(plan.dir) / STATE_FILE
    path.write_text(json.dumps(asdict(plan), indent=2) + "\n", encoding="utf-8")
    return path


def load_plan(directory: str) -> Plan:
    path = Path(directory) / STATE_FILE
    try:
        return Plan(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        raise LlamaError(f"no plan at {path}: run `python -m agent.llamacpp plan` first") from None


def fetch(plan: Plan, out: Callable[[str], None] = print, downloader=download) -> Dict[str, str]:
    """Make sure both files are here and match their sha256: what the cache restored is checked like a download."""
    Path(plan.dir).mkdir(parents=True, exist_ok=True)
    llama = plan.llama_cpp
    for attempt in (1, 2):
        if not plan.llama_archive.exists():
            out(f"Downloading llama.cpp {llama['tag']}...")
            downloader(llama["url"], plan.llama_archive, 512 * 1024 * 1024)
        try:
            verify(plan.llama_archive, llama["sha256"], f"llama.cpp {llama['tag']}")
            break
        except LlamaError:
            if attempt == 2:
                raise
            out("The cached llama.cpp didn't match its sha256: downloading it again")
    out(f"llama.cpp {llama['tag']}: sha256 {llama['sha256']} ✓")

    plan.gguf_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        if not plan.gguf_path.exists():
            out(f"Downloading {plan.repo}/{plan.file} at {plan.commit[:12]}"
                + (f" ({size_words(plan.size)})" if plan.size else "") + " from Hugging Face...")
            started = time.monotonic()
            downloader(gguf_url(plan.repo, plan.commit, plan.file), plan.gguf_path, plan.max_bytes)
            out(f"  downloaded in {time.monotonic() - started:.0f} s")
        else:
            out(f"{plan.file}: restored from the Actions cache")
        check_size(plan.gguf_path.stat().st_size, plan.max_bytes, plan.file)
        try:
            actual = verify(plan.gguf_path, plan.expected_sha256, plan.file)
            break
        except LlamaError:
            if attempt == 2:
                raise
            out(f"{plan.file} didn't match its sha256: downloading it again")
    if plan.pinned:
        out(f"{plan.file}: sha256 {actual} ✓ (matches LLAMA_GGUF_SHA256)")
    else:
        source = "matches Hugging Face's" if plan.expected_sha256 else "Hugging Face didn't publish one"
        out(f"{plan.file}: sha256 {actual} ({source}). Pin it: set the repository variable "
            f"LLAMA_GGUF_SHA256={actual}")
    _write_stamp(plan, actual)
    extract_llama(plan.llama_archive, Path(plan.dir) / "llama")
    return {"gguf_sha256": actual}


# --- llama-server -----------------------------------------------------------------------------------------------------

def http_get(url: str, timeout: float = 10) -> Tuple[Optional[int], Any]:
    """(status, parsed JSON or None); (None, None) when nothing answered."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as e:
        e.close()
        return e.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return None, None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def server_command(plan: Plan, bin_dir: Path) -> List[str]:
    return [str(bin_dir / "llama-server"), "-m", str(plan.gguf_path), "--alias", plan.alias,
            "--host", "127.0.0.1", "--port", str(plan.port), "-c", str(plan.ctx), "--parallel", "1", "--jinja",
            *plan.server_args]


def server_env(bin_dir: Path, environ: Mapping[str, str] = os.environ) -> Dict[str, str]:
    """llama-server's environment: no secrets and no way to ask GitHub for tokens."""
    keep = {k: v for k, v in environ.items()
            if not k.startswith("ACTIONS_") and not k.endswith(("_API_KEY", "_TOKEN")) and k != "OPENAI_BASE_URL"}
    keep["LD_LIBRARY_PATH"] = str(bin_dir) + (":" + keep["LD_LIBRARY_PATH"] if keep.get("LD_LIBRARY_PATH") else "")
    return keep


def wait_for_health(base: str, process: Any, timeout: float = SERVER_START_SECONDS, *, get=http_get,
                    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> float:
    """Poll /health until llama-server has loaded the model (it answers 503 while loading). Returns the seconds it took.
    Fails fast if the process dies."""
    started = clock()
    while True:
        code = process.poll()
        if code is not None:
            raise LlamaError(f"llama-server stopped (exit code {code}) before it was ready")
        status, _ = get(f"{base}/health", 5)
        if status == 200:
            return clock() - started
        if clock() - started >= timeout:
            raise LlamaError(f"llama-server wasn't ready after {timeout / 60:.0f} minutes")
        sleep(2)


def build_info(base: str, get=http_get) -> Optional[str]:
    """llama-server's own build string ("b12387-a956a852c"), from /props."""
    status, data = get(f"{base}/props", 10)
    value = data.get("build_info") if status == 200 and isinstance(data, dict) else None
    return str(value)[:100] if isinstance(value, str) and value else None


def machine_words() -> str:
    """'ubuntu 24.04, 4 vCPU, 15.6 GiB', as the League's records say it."""
    name = platform.system()
    try:
        release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
        name = f"{release.get('ID', name)} {release.get('VERSION_ID', '').strip(chr(34))}".strip()
    except OSError:
        pass
    memory = ""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory = f", {int(line.split()[1]) / 1024 / 1024:.1f} GiB"
    except (OSError, ValueError, IndexError):
        pass
    return f"{name}, {os.cpu_count()} vCPU{memory}"[:200]


def provenance(plan: Plan, gguf_sha256: str, build: Optional[str], environ: Mapping[str, str] = os.environ
               ) -> Dict[str, Any]:
    """What ran, for the public run log: the exact file, the llama.cpp build and the Actions run."""
    llama = plan.llama_cpp
    return {"gguf_repo": plan.repo, "gguf_file": plan.file, "gguf_revision": plan.commit, "gguf_sha256": gguf_sha256,
            "gguf_size": plan.size, "llama_cpp": build or f"{llama['tag']}-{llama['commit'][:9]}",
            "llama_cpp_sha256": llama["sha256"], "machine": machine_words(), "run_url": run_url(environ)}


def start(plan: Plan, out: Callable[[str], None] = print, popen=subprocess.Popen, get=http_get,
          timeout: float = SERVER_START_SECONDS) -> Dict[str, Any]:
    bin_dir = Path(plan.dir) / "llama"
    found = sorted(bin_dir.rglob("llama-server"), key=lambda p: len(p.parts)) if bin_dir.exists() else []
    if not found:
        raise LlamaError("llama.cpp isn't unpacked: run `python -m agent.llamacpp fetch` first")
    bin_dir = found[0].parent
    # fetch() checked the file; check it again only if it changed since (it's 3 GB: hashing takes a while)
    gguf_sha256 = verify_once(plan)
    log_path = Path(plan.dir) / SERVER_LOG
    out(f"Starting llama-server: {plan.alias}, context {plan.ctx}, on 127.0.0.1:{plan.port}")
    with open(log_path, "wb") as log:
        # Its own session, output to a file: the server outlives this step, and the step's pipes don't wait for it
        process = popen(server_command(plan, bin_dir), stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, env=server_env(bin_dir), start_new_session=True)
    (Path(plan.dir) / PID_FILE).write_text(str(process.pid), encoding="utf-8")
    base = f"http://127.0.0.1:{plan.port}"
    try:
        seconds = wait_for_health(base, process, timeout, get=get)
    except LlamaError:
        # No prompt has reached the server yet, so its log is safe to show
        out("llama-server's last lines:")
        for line in tail(log_path):
            out(f"  {line}")
        stop(plan.dir, out=lambda _: None)
        raise
    build = build_info(base, get)
    record = provenance(plan, gguf_sha256, build)
    (Path(plan.dir) / PROVENANCE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    out(f"llama-server is ready after {seconds:.0f} s (llama.cpp {record['llama_cpp']}, {record['machine']})")
    return record


def _stamp_path(plan: Plan) -> Path:
    return plan.gguf_path.with_name(plan.gguf_path.name + ".sha256")


def _mark(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _write_stamp(plan: Plan, sha256: str) -> None:
    _stamp_path(plan).write_text(f"{_mark(plan.gguf_path)} {sha256}\n", encoding="utf-8")


def verify_once(plan: Plan) -> str:
    """The GGUF's sha256: fetch() leaves a stamp beside it, so start() doesn't hash 3 GB twice. A file that changed
    since (another size or time) is hashed again."""
    try:
        saved_mark, saved_sha = _stamp_path(plan).read_text(encoding="utf-8").split()
        if saved_mark == _mark(plan.gguf_path) and (not plan.expected_sha256 or saved_sha == plan.expected_sha256):
            return saved_sha
    except (OSError, ValueError):
        pass
    actual = verify(plan.gguf_path, plan.expected_sha256, plan.file)
    _write_stamp(plan, actual)
    return actual


def tail(path: Path, lines: int = 40) -> List[str]:
    try:
        return [line[:300] for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]]
    except OSError:
        return []


def _is_llama_server(pid: int) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.parent.exists():
        return True  # no /proc (macOS): trust the pid file
    try:
        return b"llama-server" in cmdline.read_bytes()
    except OSError:
        return False


def stop(directory: str, out: Callable[[str], None] = print, kill=os.kill, sleep=time.sleep,
         is_server: Callable[[int], bool] = _is_llama_server) -> bool:
    """Stop the llama-server that start() left running. True if one was stopped."""
    pid_file = Path(directory) / PID_FILE
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        out("No llama-server to stop")
        return False
    pid_file.unlink(missing_ok=True)
    if not is_server(pid):
        out("llama-server had already stopped")
        return False
    try:
        kill(pid, signal.SIGTERM)
        for _ in range(20):
            sleep(0.5)
            kill(pid, 0)  # raises once it's gone
        kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        out("couldn't stop llama-server (permission denied)")
        return False
    out("llama-server stopped")
    return True


def load_provenance(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# --- Command line (the workflow's steps) ------------------------------------------------------------------------------

def _outputs(values: Mapping[str, Any]) -> None:
    """Step outputs for the workflow (GITHUB_OUTPUT), or printed when run by hand."""
    lines = [f"{key}={value}" for key, value in values.items()]
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        for line in lines:
            print(line)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.llamacpp", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan_cmd = sub.add_parser("plan", help="check the settings and the file's size; pick a folder; print cache keys")
    plan_cmd.add_argument("--dirs", nargs="+", default=[str(Path.home() / ".cache" / "arena-starter-llamacpp")])
    for name in ("fetch", "start", "stop"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--dir", default=os.environ.get("LLAMA_DIR"), help="the folder `plan` picked (LLAMA_DIR)")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            s = settings()
            plan = make_plan(s, args.dirs)
            save_plan(plan)
            which = "the default model (the AI League's Qwen3.5 4B)" if s.is_default else "your model"
            print(f"llama.cpp mode: {which}: {plan.repo}/{plan.file}")
            print(f"  revision {plan.revision}" + (f" = commit {plan.commit}" if plan.revision != plan.commit else ""))
            print(f"  size {size_words(plan.size) if plan.size else 'unknown'} (limit {size_words(plan.max_bytes)}), "
                  f"files in {plan.dir}")
            if not plan.pinned:
                print("  LLAMA_GGUF_SHA256 isn't set: this run checks the file against Hugging Face's sha256 and "
                      "prints the one it computed")
            _outputs({"dir": plan.dir, "gguf_path": plan.gguf_path, "gguf_key": plan.gguf_cache_key,
                      "llama_path": plan.llama_archive, "llama_key": plan.llama_cache_key,
                      "provenance": Path(plan.dir) / PROVENANCE_FILE})
            return 0
        if not args.dir:
            raise LlamaError("--dir (or LLAMA_DIR) is required")
        if args.command == "stop":
            stop(args.dir)
            return 0
        plan = load_plan(args.dir)
        if args.command == "fetch":
            fetch(plan)
            return 0
        start(plan)
        return 0
    except ConfigError as e:  # LlamaError, or a number setting out of range
        print(f"::error title=llama.cpp::{e}" if os.environ.get("GITHUB_ACTIONS") == "true" else f"Error: {e}",
              flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
