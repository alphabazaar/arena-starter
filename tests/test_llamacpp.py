"""llama.cpp on the runner (agent/llamacpp.py): settings, the size guard, sha256 checks, downloads, starting and stopping
llama-server, and a whole run against it. Everything is faked: no model is downloaded and no server is started."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import signal
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import llamacpp, main as agent_main, providers
from agent.config import ConfigError
from agent.llamacpp import GB, LlamaError
from tests.fakes import TEST_KEY, FakeServer, fake_ab

SHA_A = "a" * 64
COMMIT = "1234567890abcdef1234567890abcdef12345678"


def env(**values):
    return mock.patch.dict(os.environ, values, clear=True)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Settings(unittest.TestCase):
    def test_no_settings_means_the_pinned_league_model(self):
        with env():
            s = llamacpp.settings()
        self.assertTrue(s.is_default)
        self.assertEqual((s.repo, s.file, s.revision, s.sha256), (
            "ggml-org/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", "8d43d974efc48399cb58957ad49c8b26bf5a109c",
            "217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14"))
        self.assertEqual((s.alias, s.base_url), ("Qwen3.5-4B-Q4_K_M", "http://127.0.0.1:8080/v1"))
        self.assertEqual((s.ctx, s.max_tokens, s.timeout, s.max_bytes), (24576, 8192, 3300, int(12.5 * GB)))
        self.assertEqual(s.server_args, ())

    def test_your_own_model(self):
        with env(LLAMA_GGUF_REPO="https://huggingface.co/bartowski/Some-Model-GGUF/", LLAMA_GGUF_FILE="Some-Q4_K_M.gguf",
                 LLAMA_GGUF_SHA256=SHA_A.upper(), LLAMA_CTX_SIZE="8192", LLAMA_MAX_TOKENS="2048",
                 LLAMA_TIMEOUT_MINUTES="10", LLAMA_SERVER_ARGS="--reasoning-budget 0 -t 4",
                 LLAMA_MAX_GGUF_GB="6"):
            s = llamacpp.settings()
        self.assertFalse(s.is_default)
        self.assertEqual((s.repo, s.file, s.revision, s.sha256), ("bartowski/Some-Model-GGUF", "Some-Q4_K_M.gguf",
                                                                  "main", SHA_A))
        self.assertEqual((s.ctx, s.max_tokens, s.timeout, s.max_bytes), (8192, 2048, 600, 6 * GB))
        self.assertEqual(s.server_args, ("--reasoning-budget", "0", "-t", "4"))

    def test_sha256_is_optional(self):
        with env(LLAMA_GGUF_REPO="org/repo", LLAMA_GGUF_FILE="m.gguf", LLAMA_GGUF_REVISION=COMMIT):
            s = llamacpp.settings()
        self.assertEqual((s.sha256, s.revision), (None, COMMIT))

    def test_bad_settings_say_which_one(self):
        cases = [
            ({"LLAMA_GGUF_FILE": "m.gguf"}, "LLAMA_GGUF_REPO"),
            ({"LLAMA_GGUF_SHA256": SHA_A}, "LLAMA_GGUF_REPO"),
            ({"LLAMA_GGUF_REPO": "org/repo"}, "LLAMA_GGUF_FILE"),
            ({"LLAMA_GGUF_REPO": "not a repo", "LLAMA_GGUF_FILE": "m.gguf"}, "LLAMA_GGUF_REPO must be"),
            ({"LLAMA_GGUF_REPO": "org/repo", "LLAMA_GGUF_FILE": "model.bin"}, "LLAMA_GGUF_FILE must be"),
            ({"LLAMA_GGUF_REPO": "org/repo", "LLAMA_GGUF_FILE": "../x.gguf"}, "LLAMA_GGUF_FILE must be"),
            ({"LLAMA_GGUF_REPO": "org/repo", "LLAMA_GGUF_FILE": "m-00001-of-00003.gguf"}, "split GGUF"),
            ({"LLAMA_GGUF_REPO": "org/repo", "LLAMA_GGUF_FILE": "m.gguf", "LLAMA_GGUF_REVISION": "a b"},
             "LLAMA_GGUF_REVISION"),
            ({"LLAMA_GGUF_REPO": "org/repo", "LLAMA_GGUF_FILE": "m.gguf", "LLAMA_GGUF_SHA256": "abc"},
             "64 hex"),
            ({"LLAMA_SERVER_ARGS": "--port 9999"}, "can't set --port"),
            ({"LLAMA_SERVER_ARGS": "--host=0.0.0.0"}, "can't set --host"),
            ({"LLAMA_SERVER_ARGS": "-hf org/other"}, "can't set -hf"),
            ({"LLAMA_SERVER_ARGS": "-hfd org/draft"}, "can't set -hfd"),
            ({"LLAMA_SERVER_ARGS": "--hf-repo-draft=draft"}, "can't set --hf-repo-draft"),
            ({"LLAMA_SERVER_ARGS": "-hft hf_x"}, "can't set -hft"),
            ({"LLAMA_SERVER_ARGS": "--mmproj-url https://x/y.gguf"}, "can't set --mmproj-url"),
            ({"LLAMA_SERVER_ARGS": "-np 4"}, "can't set -np"),
            ({"LLAMA_TIMEOUT_MINUTES": "70"}, "LLAMA_TIMEOUT_MINUTES must be from 1 to 65"),
            ({"LLAMA_SERVER_ARGS": "--x $(rm)"}, "plain values"),
            ({"LLAMA_MAX_TOKENS": "lots"}, "LLAMA_MAX_TOKENS must be a number"),
            ({"LLAMA_CTX_SIZE": "100"}, "LLAMA_CTX_SIZE must be from"),
        ]
        for values, message in cases:
            with self.subTest(values=values), env(**values), self.assertRaisesRegex(ConfigError, message):
                llamacpp.settings()

    def test_errors_are_config_errors(self):
        self.assertTrue(issubclass(LlamaError, ConfigError))  # main() turns them into "setup: ..." and exit code 2


class SizeGuard(unittest.TestCase):
    def test_refuses_a_file_too_big_for_the_runner(self):
        with self.assertRaisesRegex(LlamaError, r"big-model\.gguf is 13\.0 GB, more than the 12\.0 GB.*16 GB"):
            llamacpp.check_size(13 * GB, 12 * GB, "big-model.gguf")

    def test_allows_what_fits_or_isnt_known(self):
        llamacpp.check_size(3 * GB, 12 * GB, "small.gguf")
        llamacpp.check_size(12 * GB, 12 * GB, "exact.gguf")
        llamacpp.check_size(None, 12 * GB, "unknown.gguf")


def head_answer(status=302, commit=COMMIT, size=3 * GB, etag=SHA_A):
    headers = {"X-Repo-Commit": commit, "X-Linked-Size": str(size), "X-Linked-Etag": f'"{etag}"'}
    calls = []

    def head(url):
        calls.append(url)
        return status, {k.lower(): v for k, v in headers.items() if v is not None}
    return head, calls


class Remote(unittest.TestCase):
    def settings(self, **values):
        with env(LLAMA_GGUF_REPO="org/repo", LLAMA_GGUF_FILE="m.gguf", **values):
            return llamacpp.settings()

    def test_reads_commit_size_and_sha256_from_hugging_face(self):
        head, calls = head_answer()
        remote = llamacpp.describe_remote(self.settings(), head)
        self.assertEqual((remote.commit, remote.size, remote.sha256), (COMMIT, 3 * GB, SHA_A))
        self.assertEqual(calls, ["https://huggingface.co/org/repo/resolve/main/m.gguf"])

    def test_gated_and_missing_files(self):
        for status, message in ((401, "private or gated"), (403, "private or gated"), (404, "has no m.gguf")):
            with self.subTest(status=status), self.assertRaisesRegex(LlamaError, message):
                llamacpp.describe_remote(self.settings(), head_answer(status=status)[0])

    def test_plan_downloads_the_resolved_commit_and_keys_the_cache_by_sha256(self):
        head, _ = head_answer()
        folder = tempfile.mkdtemp()
        plan = llamacpp.make_plan(self.settings(), [folder], head, free=lambda d: 100 * GB)
        self.assertEqual((plan.commit, plan.expected_sha256, plan.pinned), (COMMIT, SHA_A, False))
        self.assertEqual(plan.gguf_cache_key, f"gguf-{SHA_A}")
        self.assertEqual(plan.gguf_path, Path(folder) / "models" / "m.gguf")
        self.assertTrue(plan.llama_cache_key.startswith("llamacpp-b12387-"))

    def test_plan_refuses_a_pinned_sha256_that_hugging_face_contradicts(self):
        head, _ = head_answer(etag="b" * 64)
        with self.assertRaisesRegex(LlamaError, "nothing was downloaded"):
            llamacpp.make_plan(self.settings(LLAMA_GGUF_SHA256=SHA_A), [tempfile.mkdtemp()], head,
                               free=lambda d: 100 * GB)

    def test_plan_applies_the_size_guard_before_downloading(self):
        head, _ = head_answer(size=20 * GB)
        with self.assertRaisesRegex(LlamaError, "20.0 GB"):
            llamacpp.make_plan(self.settings(), [tempfile.mkdtemp()], head, free=lambda d: 100 * GB)

    def test_plan_needs_disk_room(self):
        head, _ = head_answer()
        with self.assertRaisesRegex(LlamaError, "no disk has room"):
            llamacpp.make_plan(self.settings(), [tempfile.mkdtemp()], head, free=lambda d: 1 * GB)

    def test_without_any_sha256_the_cache_key_is_the_commit(self):
        head, _ = head_answer(etag="not-a-sha")
        plan = llamacpp.make_plan(self.settings(), [tempfile.mkdtemp()], head, free=lambda d: 100 * GB)
        self.assertIsNone(plan.expected_sha256)
        self.assertEqual(plan.gguf_cache_key, f"gguf-org-repo-{COMMIT}-m.gguf")


class Sha256(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "m.gguf"
        self.path.write_bytes(b"GGUF fake weights")

    def test_matching_file(self):
        self.assertEqual(llamacpp.verify(self.path, sha(b"GGUF fake weights"), "m.gguf"), sha(b"GGUF fake weights"))

    def test_no_expected_sha256_just_computes_it(self):
        self.assertEqual(llamacpp.verify(self.path, None, "m.gguf"), sha(b"GGUF fake weights"))

    def test_mismatch_deletes_the_file(self):
        with self.assertRaisesRegex(LlamaError, "sha256 mismatch.*nothing ran"):
            llamacpp.verify(self.path, SHA_A, "m.gguf")
        self.assertFalse(self.path.exists())

    def test_hashes_in_chunks(self):
        self.assertEqual(llamacpp.sha256_file(self.path, chunk=4), sha(b"GGUF fake weights"))


class Download(unittest.TestCase):
    def serve(self, routes):
        server = FakeServer(routes)
        self.addCleanup(server.close)
        return server

    def test_downloads_anonymously(self):
        server = self.serve({("GET", "/m.gguf"): (200, b"x" * 5000)})
        dest = Path(tempfile.mkdtemp()) / "m.gguf"
        self.assertEqual(llamacpp.download(server.url + "/m.gguf", dest, 10_000), 5000)
        self.assertEqual(dest.read_bytes(), b"x" * 5000)
        self.assertFalse(dest.with_name("m.gguf.part").exists())
        self.assertNotIn("authorization", server.requests[0]["headers"])

    def test_stops_at_the_size_limit(self):
        server = self.serve({("GET", "/m.gguf"): (200, b"x" * 5000)})
        dest = Path(tempfile.mkdtemp()) / "m.gguf"
        with self.assertRaisesRegex(LlamaError, "bigger than"):
            llamacpp.download(server.url + "/m.gguf", dest, 1000)
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_name("m.gguf.part").exists())

    def test_retries_server_errors_then_gives_up(self):
        server = self.serve({("GET", "/m.gguf"): [(503, {"error": "busy"}), (200, b"ok")]})
        dest = Path(tempfile.mkdtemp()) / "m.gguf"
        llamacpp.download(server.url + "/m.gguf", dest, 1000, sleep=lambda s: None)
        self.assertEqual(dest.read_bytes(), b"ok")
        server = self.serve({("GET", "/m.gguf"): (503, {"error": "busy"})})
        with self.assertRaisesRegex(LlamaError, "couldn't download"):
            llamacpp.download(server.url + "/m.gguf", dest, 1000, retries=2, sleep=lambda s: None)
        self.assertEqual(len(server.requests), 3)

    def test_a_missing_file_isnt_retried(self):
        server = self.serve({})
        with self.assertRaisesRegex(LlamaError, "answered 404"):
            llamacpp.download(server.url + "/m.gguf", Path(tempfile.mkdtemp()) / "m.gguf", 1000,
                              sleep=lambda s: None)
        self.assertEqual(len(server.requests), 1)


def llama_archive(path: Path) -> bytes:
    """A stand-in for the llama.cpp release: a folder with a llama-server and a library."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in (("llama-b12387/llama-server", b"#!/bin/sh\n"), ("llama-b12387/libllama.so", b"lib")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return path.read_bytes()


class Fetch(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.weights = b"GGUF the real weights"
        archive = llama_archive(self.dir / "built.tar.gz")
        self.files = {"llama": archive, "gguf": self.weights}
        self.downloads = []
        llama = dict(llamacpp.LLAMA_CPP, sha256=sha(archive))
        self.plan = llamacpp.Plan(dir=str(self.dir), repo="org/repo", file="m.gguf", revision="main", commit=COMMIT,
                                  size=len(self.weights), expected_sha256=sha(self.weights), pinned=True,
                                  alias="m", ctx=4096, port=8080, server_args=[], max_bytes=GB, llama_cpp=llama)

    def downloader(self, url, dest, max_bytes):
        self.downloads.append(url)
        dest.write_bytes(self.files["llama" if url.endswith(".tar.gz") else "gguf"])

    def fetch(self):
        out = []
        result = llamacpp.fetch(self.plan, out=out.append, downloader=self.downloader)
        return result, "\n".join(out)

    def test_downloads_checks_and_unpacks(self):
        result, out = self.fetch()
        self.assertEqual(result["gguf_sha256"], sha(self.weights))
        self.assertEqual(self.downloads, [self.plan.llama_cpp["url"],
                                          f"https://huggingface.co/org/repo/resolve/{COMMIT}/m.gguf"])
        self.assertTrue((self.dir / "llama" / "llama-b12387" / "llama-server").is_file())
        self.assertIn("matches LLAMA_GGUF_SHA256", out)
        self.assertEqual(llamacpp.verify_once(self.plan), sha(self.weights))  # the stamp fetch left

    def test_a_cache_hit_is_checked_too(self):
        self.plan.llama_archive.write_bytes(self.files["llama"])
        self.plan.gguf_path.parent.mkdir(parents=True)
        self.plan.gguf_path.write_bytes(self.weights)
        _, out = self.fetch()
        self.assertEqual(self.downloads, [])
        self.assertIn("restored from the Actions cache", out)

    def test_a_bad_cached_file_is_downloaded_again(self):
        self.plan.gguf_path.parent.mkdir(parents=True)
        self.plan.gguf_path.write_bytes(b"tampered")
        _, out = self.fetch()
        self.assertEqual(len(self.downloads), 2)  # llama.cpp, then the GGUF again
        self.assertEqual(self.plan.gguf_path.read_bytes(), self.weights)

    def test_a_download_that_never_matches_fails(self):
        self.files["gguf"] = b"something else"
        with self.assertRaisesRegex(LlamaError, "sha256 mismatch"):
            self.fetch()
        self.assertFalse(self.plan.gguf_path.exists())

    def test_without_a_pin_the_computed_sha256_is_printed(self):
        self.plan.pinned, self.plan.expected_sha256 = False, None
        _, out = self.fetch()
        self.assertIn(f"LLAMA_GGUF_SHA256={sha(self.weights)}", out)

    def test_plan_round_trips_through_its_file(self):
        llamacpp.save_plan(self.plan)
        self.assertEqual(llamacpp.load_plan(str(self.dir)), self.plan)
        with self.assertRaisesRegex(LlamaError, "no plan"):
            llamacpp.load_plan(tempfile.mkdtemp())


class FakeProcess:
    def __init__(self, exit_after=None):
        self.pid = 4242
        self.polls = 0
        self.exit_after = exit_after
        self.waited = 0
        self.killed = False

    def wait(self, timeout=None):
        self.waited += 1
        return 1

    def kill(self):
        self.killed = True

    def poll(self):
        self.polls += 1
        return 1 if self.exit_after is not None and self.polls > self.exit_after else None


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Health(unittest.TestCase):
    def test_waits_while_the_model_loads(self):
        answers = iter([(None, None), (503, {"error": "Loading model"}), (200, {"status": "ok"})])
        clock = FakeClock()
        seconds = llamacpp.wait_for_health("http://127.0.0.1:8080", FakeProcess(), 60,
                                           get=lambda url, timeout: next(answers), clock=clock, sleep=clock.sleep)
        self.assertEqual(seconds, 4)

    def test_gives_up_after_the_timeout(self):
        clock = FakeClock()
        with self.assertRaisesRegex(LlamaError, "wasn't ready after 1 minutes"):
            llamacpp.wait_for_health("http://x", FakeProcess(), 60, get=lambda url, timeout: (503, None),
                                     clock=clock, sleep=clock.sleep)

    def test_fails_fast_when_the_server_dies(self):
        clock = FakeClock()
        with self.assertRaisesRegex(LlamaError, "exit code 1"):
            llamacpp.wait_for_health("http://x", FakeProcess(exit_after=2), 600, get=lambda url, timeout: (503, None),
                                     clock=clock, sleep=clock.sleep)
        self.assertLess(clock.now, 10)

    def test_build_info(self):
        self.assertEqual(llamacpp.build_info("http://x", lambda url, t: (200, {"build_info": "b12387-a956a852c"})),
                         "b12387-a956a852c")
        self.assertIsNone(llamacpp.build_info("http://x", lambda url, t: (None, None)))

    def test_http_get_against_a_server(self):
        server = FakeServer({("GET", "/health"): (503, {"error": "loading"}), ("GET", "/props"): (200, {"a": 1})})
        self.addCleanup(server.close)
        self.assertEqual(llamacpp.http_get(server.url + "/health"), (503, None))
        self.assertEqual(llamacpp.http_get(server.url + "/props"), (200, {"a": 1}))
        self.assertEqual(llamacpp.http_get("http://127.0.0.1:9/nothing", 1), (None, None))


class StartStop(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "llama" / "llama-b12387").mkdir(parents=True)
        (self.dir / "llama" / "llama-b12387" / "llama-server").write_bytes(b"")
        (self.dir / "models").mkdir()
        self.weights = b"GGUF weights"
        (self.dir / "models" / "m.gguf").write_bytes(self.weights)
        self.plan = llamacpp.Plan(dir=str(self.dir), repo="org/repo", file="m.gguf", revision="main", commit=COMMIT,
                                  size=12, expected_sha256=sha(self.weights), pinned=True, alias="m", ctx=4096,
                                  port=8099, server_args=["--reasoning-budget", "0"], max_bytes=GB)
        self.started = []

    def popen(self, process):
        def fake(command, **kwargs):
            self.started.append((command, kwargs))
            return process
        return fake

    def test_starts_on_localhost_with_the_checked_file_and_records_provenance(self):
        get = {"http://127.0.0.1:8099/health": (200, {"status": "ok"}),
               "http://127.0.0.1:8099/props": (200, {"build_info": "b12387-a956a852c"})}
        environment = {"GITHUB_RUN_ID": "99", "GITHUB_REPOSITORY": "me/my-agent", "AB_API_KEY": TEST_KEY,
                       "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "t", "PATH": "/usr/bin"}
        with env(**environment):
            record = llamacpp.start(self.plan, out=lambda line: None, popen=self.popen(FakeProcess()),
                                    get=lambda url, timeout: get[url])
        [(command, kwargs)] = self.started
        self.assertEqual(command[1:], ["-m", str(self.dir / "models" / "m.gguf"), "--alias", "m", "--host",
                                       "127.0.0.1", "--port", "8099", "-c", "4096", "--parallel", "1", "--jinja",
                                       "--reasoning-budget", "0"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn("AB_API_KEY", kwargs["env"])  # llama-server never sees a secret
        self.assertNotIn("ACTIONS_ID_TOKEN_REQUEST_TOKEN", kwargs["env"])
        self.assertTrue(kwargs["env"]["LD_LIBRARY_PATH"].endswith("llama-b12387"))
        self.assertEqual((self.dir / "llama-server.pid").read_text(), "4242")
        self.assertEqual({k: record[k] for k in ("gguf_repo", "gguf_file", "gguf_revision", "gguf_sha256",
                                                 "llama_cpp", "run_url")},
                         {"gguf_repo": "org/repo", "gguf_file": "m.gguf", "gguf_revision": COMMIT,
                          "gguf_sha256": sha(self.weights), "llama_cpp": "b12387-a956a852c",
                          "run_url": "https://github.com/me/my-agent/actions/runs/99"})
        self.assertEqual(record["server_args"], "--reasoning-budget 0")
        self.assertEqual(json.loads((self.dir / "provenance.json").read_text()), record)

    def test_a_server_that_dies_shows_its_log_and_fails(self):
        (self.dir / "llama-server.log").write_text("")
        out = []

        process = FakeProcess(exit_after=0)

        def popen(command, stdout, **kwargs):
            stdout.write(b"error: failed to load model\n")
            return process
        with env(), mock.patch.object(llamacpp, "stop") as stop, self.assertRaisesRegex(LlamaError, "exit code 1"):
            llamacpp.start(self.plan, out=out.append, popen=popen, get=lambda url, timeout: (None, None))
        self.assertIn("  error: failed to load model", out)
        stop.assert_called_once()
        self.assertGreaterEqual(process.waited, 1)  # reaped, no zombie left behind

    def test_a_server_that_never_gets_ready_is_stopped_and_reaped(self):
        process = FakeProcess()
        with env(), mock.patch.object(llamacpp, "stop") as stop, \
                self.assertRaisesRegex(LlamaError, "wasn't ready"):
            llamacpp.start(self.plan, out=lambda line: None, popen=self.popen(process),
                           get=lambda url, timeout: (503, None), timeout=0)
        stop.assert_called_once()
        self.assertEqual(process.waited, 1)

    def test_a_stale_stamp_means_hashing_again(self):
        gguf = self.dir / "models" / "m.gguf"
        stamp = self.dir / "models" / "m.gguf.sha256"
        stamp.write_text(f"1:2 {sha(self.weights)}\n")  # another size and time: the file changed since
        self.assertEqual(llamacpp.verify_once(self.plan), sha(self.weights))
        self.assertTrue(stamp.read_text().startswith(f"{gguf.stat().st_size}:"))
        stat = gguf.stat()
        stamp.write_text(f"{stat.st_size}:{stat.st_mtime_ns} {SHA_A}\n")  # right file, but not the expected sha256
        self.assertEqual(llamacpp.verify_once(self.plan), sha(self.weights))
        gguf.write_bytes(b"swapped weights!")
        with self.assertRaisesRegex(LlamaError, "sha256 mismatch"):
            llamacpp.verify_once(self.plan)

    def test_refuses_a_file_that_changed_since_it_was_checked(self):
        (self.dir / "models" / "m.gguf").write_bytes(b"swapped")
        with env(), self.assertRaisesRegex(LlamaError, "sha256 mismatch"):
            llamacpp.start(self.plan, out=lambda line: None, popen=self.popen(FakeProcess()),
                           get=lambda url, timeout: (200, {}))
        self.assertEqual(self.started, [])

    def test_stop_terminates_the_server(self):
        (self.dir / "llama-server.pid").write_text("4242")
        sent = []

        def kill(pid, sig):
            sent.append((pid, sig))
            if sig == 0:
                raise ProcessLookupError
        self.assertTrue(llamacpp.stop(str(self.dir), out=lambda line: None, kill=kill, sleep=lambda s: None,
                                      is_server=lambda pid: True))
        self.assertEqual(sent, [(4242, signal.SIGTERM), (4242, 0)])
        self.assertFalse((self.dir / "llama-server.pid").exists())

    def test_stop_kills_a_server_that_ignores_sigterm(self):
        (self.dir / "llama-server.pid").write_text("4242")
        sent = []
        llamacpp.stop(str(self.dir), out=lambda line: None, kill=lambda pid, sig: sent.append(sig),
                      sleep=lambda s: None, is_server=lambda pid: True)
        self.assertEqual((sent[0], sent[-1]), (signal.SIGTERM, signal.SIGKILL))

    def test_stop_leaves_other_processes_alone(self):
        (self.dir / "llama-server.pid").write_text("4242")
        kill = mock.Mock()
        self.assertFalse(llamacpp.stop(str(self.dir), out=lambda line: None, kill=kill, is_server=lambda pid: False))
        kill.assert_not_called()
        self.assertFalse(llamacpp.stop(str(self.dir), out=lambda line: None, kill=kill))  # no pid file now


class CommandLine(unittest.TestCase):
    def test_plan_writes_step_outputs(self):
        folder = tempfile.mkdtemp()
        outputs = Path(tempfile.mkdtemp()) / "out"
        head = head_answer(commit="8d43d974efc48399cb58957ad49c8b26bf5a109c", size=3032091584,
                           etag="217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14")[0]
        with env(GITHUB_OUTPUT=str(outputs)), mock.patch.object(llamacpp, "http_head", head), \
                mock.patch.object(llamacpp.shutil, "disk_usage", return_value=mock.Mock(free=100 * GB)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(llamacpp.main(["plan", "--dirs", folder]), 0)
        values = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
        self.assertEqual(values["gguf_key"], "gguf-217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14")
        self.assertEqual(values["dir"], folder)
        self.assertIn("the default model", out.getvalue())

    def test_errors_become_annotations(self):
        with env(GITHUB_ACTIONS="true", LLAMA_GGUF_FILE="m.gguf"), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(llamacpp.main(["plan", "--dirs", tempfile.mkdtemp()]), 2)
        self.assertIn("::error title=llama.cpp::set both LLAMA_GGUF_REPO", out.getvalue())


def installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def completion(content, finish_reason="stop"):
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1760000000, "model": "Qwen3.5-4B-Q4_K_M",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                                                 "reasoning_content": "Let me think about NIC..."},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 7012, "completion_tokens": 2950, "total_tokens": 9962}}


class Provider(unittest.TestCase):
    def test_choose_llamacpp(self):
        for alias in ("llamacpp", "llama.cpp", "gguf"):
            with self.subTest(alias=alias), env(LLM_PROVIDER=alias, ANTHROPIC_API_KEY="a"):
                provider = providers.choose()
            self.assertEqual((provider.name, provider.label, provider.base_url),
                             ("llamacpp", "Qwen3.5-4B-Q4_K_M", "http://127.0.0.1:8080/v1"))
            self.assertEqual((provider.retries, provider.max_tokens, provider.json_mode), (0, 8192, False))
            self.assertIsNone(provider.provenance)

    def test_choose_reads_the_provenance_file(self):
        path = Path(tempfile.mkdtemp()) / "provenance.json"
        path.write_text(json.dumps({"gguf_sha256": SHA_A}))
        with env(LLM_PROVIDER="llamacpp", LLAMA_PROVENANCE=str(path)):
            self.assertEqual(providers.choose().provenance, {"gguf_sha256": SHA_A})

    def test_bad_settings_stop_the_run(self):
        with env(LLM_PROVIDER="llamacpp", LLAMA_GGUF_REPO="org/repo"), self.assertRaises(ConfigError):
            providers.choose()

    @unittest.skipUnless(installed("openai"), "openai isn't installed")
    def test_request_shape(self):
        decision = {"action": "buy", "ticker": "NVDA", "time_horizon": "1W", "confidence": 6, "stop_loss_pct": 6,
                    "take_profit_pct": 12, "reasoning": "Congress and 13F funds agree."}
        server = FakeServer({("POST", "/v1/chat/completions"): (200, completion(json.dumps(decision)))})
        self.addCleanup(server.close)
        with env(LLAMA_PORT=str(server.httpd.server_address[1])):
            provider = providers.LlamaCpp(llamacpp.settings())
        self.assertEqual(json.loads(provider.decide("system", "user")), decision)
        body = server.requests[0]["body"]
        self.assertEqual((body["model"], body["max_tokens"]), ("Qwen3.5-4B-Q4_K_M", 8192))
        self.assertNotIn("response_format", body)  # the model thinks first, as in the AI League
        self.assertEqual(provider.stats, {"prompt_tokens": 7012, "completion_tokens": 2950})
        self.assertIsNotNone(provider.seconds)

    @unittest.skipUnless(installed("openai"), "openai isn't installed")
    def test_cut_off_names_the_budget(self):
        server = FakeServer({("POST", "/v1/chat/completions"): (200, completion("", "length"))})
        self.addCleanup(server.close)
        with env(LLAMA_PORT=str(server.httpd.server_address[1]), LLAMA_MAX_TOKENS="512"):
            provider = providers.LlamaCpp(llamacpp.settings())
        with self.assertRaisesRegex(providers.ProviderError, "cut off at 512 tokens"):
            provider.decide("system", "user")
        self.assertEqual(len(server.requests), 1)  # never asked twice


@unittest.skipUnless(installed("openai"), "openai isn't installed")
class WholeRun(unittest.TestCase):
    """python -m agent with LLM_PROVIDER=llamacpp, against a fake Alpha Bazaar and a fake llama-server."""

    def setUp(self):
        for target in ("agent.ab.time.sleep", "agent.config.load_dotenv"):
            patcher = mock.patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.logs = Path(tempfile.mkdtemp())
        decision = {"action": "buy", "ticker": "NIC", "time_horizon": "1W", "confidence": 6, "stop_loss_pct": 8,
                    "take_profit_pct": 15, "reasoning": "NIC is up 9% since three sources agreed."}
        self.llama = FakeServer({("POST", "/v1/chat/completions"): (200, completion(json.dumps(decision)))})
        self.addCleanup(self.llama.close)
        self.ab = fake_ab()
        self.addCleanup(self.ab.close)
        self.provenance = Path(tempfile.mkdtemp()) / "provenance.json"
        self.provenance.write_text(json.dumps({
            "gguf_repo": "ggml-org/Qwen3.5-4B-GGUF", "gguf_file": "Qwen3.5-4B-Q4_K_M.gguf",
            "gguf_revision": "8d43d974efc48399cb58957ad49c8b26bf5a109c",
            "gguf_sha256": "217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14",
            "llama_cpp": "b12387-a956a852c", "run_url": "https://github.com/me/agent/actions/runs/99",
            "server_args": "--reasoning-budget 0",
            "machine": "ubuntu 24.04, 4 vCPU, 15.6 GiB"}))

    def run_agent(self, *args):
        environment = {"AB_API_BASE": self.ab.url, "AB_API_KEY": TEST_KEY, "LLM_PROVIDER": "llamacpp",
                       "LLAMA_PORT": str(self.llama.httpd.server_address[1]), "LLAMA_PROVENANCE": str(self.provenance)}
        out = io.StringIO()
        with env(**environment), contextlib.redirect_stdout(out):
            code = agent_main.main(["--log-dir", str(self.logs), *args])
        return code, out.getvalue()

    def test_publishes_and_logs_where_the_model_came_from(self):
        code, output = self.run_agent()
        self.assertEqual(code, 0, output)
        [submit] = self.ab.calls("POST", "/api/publisher/submit-pick")
        self.assertEqual((submit["body"]["ticker"], submit["body"]["action"]), ("NIC", "buy"))
        [entry] = self.ab.calls("POST", "/api/agents/me/log")
        meta = entry["body"]["meta"]
        self.assertEqual((meta["provider"], meta["model"]), ("llamacpp", "Qwen3.5-4B-Q4_K_M"))
        self.assertEqual(meta["gguf_sha256"], "217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14")
        self.assertEqual((meta["gguf_repo"], meta["gguf_revision"], meta["llama_cpp"], meta["run_url"]), (
            "ggml-org/Qwen3.5-4B-GGUF", "8d43d974efc48399cb58957ad49c8b26bf5a109c", "b12387-a956a852c",
            "https://github.com/me/agent/actions/runs/99"))
        self.assertLessEqual(len(json.dumps(meta).encode()), 2048)
        self.assertTrue(entry["body"]["content"].endswith(
            "sha256 217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14, on llama.cpp b12387-a956a852c "
            "with server flags --reasoning-budget 0 in GitHub Actions run https://github.com/me/agent/actions/runs/99."))
        self.assertEqual(meta["server_args"], "--reasoning-budget 0")
        self.assertEqual(self.ab.calls("PATCH", "/api/agents/me")[0]["body"], {"model": "Qwen3.5-4B-Q4_K_M"})
        self.assertIn("The model answered in", output)
        self.assertIn("7012 prompt tokens, 2950 written", output)

    def test_never_prints_the_prompt_or_the_key(self):
        code, output = self.run_agent("--dry-run")
        self.assertEqual(code, 0, output)
        prompt = self.llama.requests[0]["body"]["messages"][1]["content"]
        self.assertNotIn(prompt[:200], output)
        self.assertNotIn('"candidates"', output)
        self.assertNotIn(TEST_KEY, output)
        self.assertEqual(self.ab.calls("POST", "/api/agents/me/log"), [])  # a dry run writes nothing
        self.assertIn("Model file: ggml-org/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", output)


if __name__ == "__main__":
    unittest.main()
