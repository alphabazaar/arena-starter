"""register.py against a fake Alpha Bazaar API: the key lands in .env and never on screen."""
import contextlib
import io
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import register
from tests.fakes import TEST_KEY, fake_ab


def registered(request):
    name = (request["body"] or {}).get("name")
    return 200, {"success": True, "username": name, "publisher_id": "p1", "publisher_type": "ai-agent",
                 "api_key": TEST_KEY, "api_key_prefix": TEST_KEY[:12], "warning": "Shown once.",
                 "next_steps": [], "docs": {}, "arena": None}


class Register(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("agent.ab.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.env_file = Path(tempfile.mkdtemp()) / ".env"

    def register(self, server, *args, stdin_tty=False):
        argv = ["register.py", "--env-file", str(self.env_file), *args]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"AB_API_BASE": server.url}, clear=True), \
                mock.patch.object(sys, "argv", argv), mock.patch("sys.stdin.isatty", return_value=stdin_tty), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = register.main()
        return code, out.getvalue(), err.getvalue()

    def serve(self, answer=registered):
        server = fake_ab(register=answer)
        self.addCleanup(server.close)
        return server

    def test_saves_the_key_and_never_prints_it(self):
        server = self.serve()
        code, out, err = self.register(server, "--name", "My-Agent", "--accept-terms")
        self.assertEqual(code, 0, err)
        [request] = server.calls("POST", "/api/agents/register")
        self.assertEqual(request["body"], {"name": "my_agent", "accept_terms": True, "framework": "arena-starter",
                                           "model": "rules-based (no LLM)"})
        self.assertNotIn(TEST_KEY, out + err)
        saved = self.env_file.read_text()
        self.assertIn(f"AB_API_KEY={TEST_KEY}", saved)
        self.assertIn(f"AB_API_BASE={server.url}", saved)  # not production, so local runs stay on the same API
        self.assertEqual(stat.S_IMODE(self.env_file.stat().st_mode), 0o600)
        self.assertIn("Using the username 'my_agent'", out)
        self.assertIn("Key works", out)

    def test_show_key_is_opt_in(self):
        code, out, _ = self.register(self.serve(), "--name", "my_agent", "--accept-terms", "--show-key")
        self.assertEqual(code, 0)
        self.assertIn(TEST_KEY, out)

    def test_keeps_other_lines_of_an_existing_env_file(self):
        self.env_file.write_text("# mine\nGEMINI_API_KEY=g-123\n")
        code, _, err = self.register(self.serve(), "--name", "my_agent", "--accept-terms")
        self.assertEqual(code, 0, err)
        self.assertTrue(self.env_file.read_text().startswith("# mine\nGEMINI_API_KEY=g-123\nAB_API_KEY="))

    def test_taken_name_lists_alternatives_and_saves_nothing(self):
        taken = "Name 'my_agent' is already taken. Available: my_agent_ai, my_agent_bot."
        server = self.serve((400, {"detail": taken}))
        code, out, err = self.register(server, "--name", "my_agent", "--accept-terms")
        self.assertEqual(code, 1)
        self.assertIn("Available: my_agent_ai, my_agent_bot", err)
        self.assertFalse(self.env_file.exists())

    def test_refuses_to_replace_a_saved_key(self):
        self.env_file.write_text(f"AB_API_KEY={TEST_KEY}\n")
        server = self.serve()
        code, _, err = self.register(server, "--name", "my_agent", "--accept-terms")
        self.assertEqual(code, 1)
        self.assertIn("--force", err)
        self.assertEqual(server.requests, [])

    def test_bad_names_never_reach_the_api(self):
        server = self.serve()
        for name in ("ab", "starter-template-test", "émoji_agent", "a" * 21):
            with self.subTest(name=name):
                code, _, err = self.register(server, "--name", name, "--accept-terms")
                self.assertEqual(code, 1)
                self.assertIn("3-20 characters", err)
        self.assertEqual(server.requests, [])

    def test_terms_must_be_accepted(self):
        server = self.serve()
        code, _, err = self.register(server, "--name", "my_agent")
        self.assertEqual(code, 1)
        self.assertIn("--accept-terms", err)
        with mock.patch("builtins.input", return_value="n"):
            code, _, err = self.register(server, "--name", "my_agent", stdin_tty=True)
        self.assertEqual(code, 1)
        self.assertEqual(server.requests, [])
        with mock.patch("builtins.input", return_value="yes"):
            code, _, _ = self.register(server, "--name", "my_agent", stdin_tty=True)
        self.assertEqual(code, 0)

    def test_rate_limit_message(self):
        server = self.serve((429, {"detail": "Rate limit exceeded (5 per 1 hour). Try again in 3600 seconds."},
                             {"Retry-After": "3600"}))
        code, _, err = self.register(server, "--name", "my_agent", "--accept-terms")
        self.assertEqual(code, 1)
        self.assertIn("5 registrations an hour", err)
        self.assertEqual(len(server.calls("POST", "/api/agents/register")), 1)


if __name__ == "__main__":
    unittest.main()
