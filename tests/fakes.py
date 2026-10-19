"""Local stand-ins for the Alpha Bazaar API and the model APIs: canned answers and a record of every request.

Tests never touch the network or need a key.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlsplit

FIXTURES = Path(__file__).parent / "fixtures"
TEST_KEY = "ab_live_TESTKEYdo_not_print_1234567890"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# A route answers with (status, body) or (status, body, headers), a list of those (used in order, the last one
# repeats), or a function of the request that returns one.
Answer = Union[Tuple, List[Tuple], Callable[[Dict[str, Any]], Tuple]]


class FakeServer:
    def __init__(self, routes: Optional[Dict[Tuple[str, str], Answer]] = None):
        self.routes: Dict[Tuple[str, str], Answer] = dict(routes or {})
        self.requests: List[Dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def _handle(self):
                parts = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw.decode("utf-8", "replace")
                request = {"method": self.command, "path": parts.path, "query": parse_qs(parts.query),
                           "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                server.requests.append(request)
                answer = server.answer(self.command, parts.path, request)
                status, payload, headers = (answer + ({},))[:3] if len(answer) == 2 else answer
                data = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PATCH = do_DELETE = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def answer(self, method: str, path: str, request: Dict[str, Any]) -> Tuple:
        route = self.routes.get((method, path))
        if route is None:
            return 404, {"detail": f"no fake route for {method} {path}"}
        if callable(route):
            return route(request)
        if isinstance(route, list):
            return route.pop(0) if len(route) > 1 else route[0]
        return route

    def calls(self, method: Optional[str] = None, path: Optional[str] = None) -> List[Dict[str, Any]]:
        return [r for r in self.requests if (method is None or r["method"] == method)
                and (path is None or r["path"] == path)]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def fake_ab(**overrides: Answer) -> FakeServer:
    """The Alpha Bazaar API with fixture data. Override a route with fake_ab(submit=(400, {...})) style keywords."""
    picks_id = "11111111-2222-4333-8444-555555555555"

    def submit(request):
        body = request["body"] or {}
        return 200, {"status": "success", "pick_id": picks_id, "pick": dict(body, id=picks_id, status="open"),
                     "market_status": "open", "entry_price_status": "captured", "entry_price_captured": True}

    def log_entry(request):
        return 201, {"success": True, "agent": "fixture_agent", "seq": 7, "kind": "decision",
                     "content": (request["body"] or {}).get("content"), "entry_hash": "ab" * 32,
                     "prev_hash": "cd" * 32}

    my_picks = fixture("my_picks.json")

    def picks(request):
        if any(r["path"] == "/api/publisher/submit-pick" and r["method"] == "POST" for r in server.requests):
            with_new = dict(my_picks, picks=my_picks["picks"] + [
                {"id": picks_id, "ticker": "NEW", "status": "open", "position_size": 5.0,
                 "attestation_code": "AB-new123"}])
            return 200, with_new
        return 200, my_picks

    routes = {
        ("GET", "/api/agents/me"): (200, fixture("agents_me.json")),
        ("PATCH", "/api/agents/me"): (200, fixture("agents_me.json")),
        ("GET", "/api/market/pulse"): (200, fixture("pulse.json")),
        ("GET", "/api/feed/daily-digest"): (200, fixture("digest.json")),
        ("GET", "/api/confluence/smart-feed"): (200, fixture("smart_feed.json")),
        ("GET", "/api/sectors/intelligence"): (200, fixture("sectors.json")),
        ("GET", "/api/publisher/my-picks"): picks,
        ("GET", "/api/arena/me"): (200, fixture("arena_me.json")),
        ("GET", "/api/arena/seasons"): (200, fixture("seasons.json")),
        ("POST", "/api/arena/join"): (200, {"success": True, "enrolled": True}),
        ("POST", "/api/publisher/submit-pick"): submit,
        ("POST", "/api/agents/me/log"): log_entry,
    }
    names = {"me": ("GET", "/api/agents/me"), "submit": ("POST", "/api/publisher/submit-pick"),
             "log": ("POST", "/api/agents/me/log"), "picks": ("GET", "/api/publisher/my-picks"),
             "arena": ("GET", "/api/arena/me"), "sectors": ("GET", "/api/sectors/intelligence"),
             "pulse": ("GET", "/api/market/pulse"), "join": ("POST", "/api/arena/join"),
             "register": ("POST", "/api/agents/register")}
    for name, answer in overrides.items():
        routes[names[name]] = answer
    server = FakeServer(routes)
    return server
