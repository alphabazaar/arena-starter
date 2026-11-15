"""Shared by the adapter tests: run an adapter's main() against the fake Alpha Bazaar API and read what it printed.

No network, no keys, and neither framework installed: everything Alpha Bazaar answers comes from tests/fakes.py.
"""
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from tests.fakes import TEST_KEY, fake_ab, fixture

TODAY = date(2026, 11, 12)  # "today in New York" for every adapter test
SUBMIT = "/api/publisher/submit-pick"
LOG = "/api/agents/me/log"
ME = "/api/agents/me"
CLOSE_HELD = ("POST", "/api/performance/close-signal/pick-held-1")  # KGC, the open long in my_picks.json


def closed_ok(price: float = 23.81):
    return 200, {"status": "success", "action": "immediate_close", "message": "Signal closed immediately",
                 "exit_price": price, "market_status": "open", "close_reason": "manual_intraday"}


def would_send(output: str):
    """Every (method, path, body) a dry run printed as "Would send: METHOD PATH" with its JSON body below."""
    found, decoder = [], json.JSONDecoder()
    for match in re.finditer(r"Would send: (\w+) (\S+)\n?", output):
        rest = output[match.end():]
        found.append((match.group(1), match.group(2), decoder.raw_decode(rest)[0] if rest.startswith("{") else None))
    return found


class AdapterCase(unittest.TestCase):
    entry = None  # the adapter's main(), as staticmethod(main)

    def setUp(self):
        patches = [mock.patch("agent.ab.time.sleep"), mock.patch("agent.config.load_dotenv"),
                   mock.patch("adapters.common.today_in_new_york", return_value=TODAY)]
        for patcher in patches:  # no waiting, never a real .env, and a fixed date
            patcher.start()
            self.addCleanup(patcher.stop)
        self.logs = Path(tempfile.mkdtemp())
        self.tmp = Path(tempfile.mkdtemp())

    def serve(self, **overrides):
        server = fake_ab(**overrides)
        self.addCleanup(server.close)
        return server

    def run_adapter(self, server, *args, key=TEST_KEY, stdin=None, **environment):
        env = {"AB_API_BASE": server.url, **environment}
        if key:
            env["AB_API_KEY"] = key
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
            stack.enter_context(contextlib.redirect_stdout(out))
            if stdin is not None:
                stack.enter_context(mock.patch("sys.stdin", io.StringIO(stdin)))
            code = type(self).entry(["--log-dir", str(self.logs), *args])
        return code, out.getvalue()

    def write(self, data, name: str = "output.json") -> str:
        """A fixture (by file name) or any JSON, saved where the adapter can read it."""
        path = self.tmp / name
        path.write_text(json.dumps(fixture(data) if isinstance(data, str) else data), encoding="utf-8")
        return str(path)

    def log_entry(self, server):
        [entry] = server.calls("POST", LOG)
        return entry["body"]

    def assert_no_key_anywhere(self, output):
        self.assertNotIn(TEST_KEY, output)
        for file in self.logs.iterdir():
            self.assertNotIn(TEST_KEY, file.read_text(encoding="utf-8"), file.name)
