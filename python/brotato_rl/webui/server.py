"""Tiny JSON + static HTTP server for the panel (standard library only).

Listens on 127.0.0.1 only and has no authentication: anyone with access to this machine's
localhost can start runs (and, with the "custom" agent, import Python modules). Do not
expose it on a network interface.
"""
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .. import paths
from .runner import Runner, load_catalog, merge_defaults, refresh_catalog, system_info
from .schema import SCHEMA, defaults

STATIC = Path(__file__).resolve().parent / "static"
PRESETS_DIR = paths.ROOT / "python" / "presets"
_PRESET_NAME = re.compile(r"^[\w\- ]{1,64}$")


class App:
    def __init__(self):
        self.runner = Runner()
        self.catalog_lock = threading.Lock()


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BrotatoRL-Panel/1.0"

        def log_message(self, fmt, *args):  # keep the console quiet (status is polled twice a second)
            pass

        # -- helpers --------------------------------------------------------------------
        def _json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, message, status=HTTPStatus.BAD_REQUEST):
            self._json({"error": message}, status)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                raise ValueError("request too large")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data

        def _same_origin(self):
            # Blocks other web pages from driving the panel through the browser (CSRF).
            origin = self.headers.get("Origin")
            return origin is None or origin == f"http://{self.headers.get('Host')}"

        # -- routes ---------------------------------------------------------------------
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = (STATIC / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/schema":
                self._json({"schema": SCHEMA, "defaults": defaults()})
            elif path == "/api/catalog":
                self._json(load_catalog())
            elif path == "/api/system":
                self._json(system_info())
            elif path == "/api/status":
                self._json(app.runner.snapshot())
            elif path == "/api/presets":
                PRESETS_DIR.mkdir(parents=True, exist_ok=True)
                self._json(sorted(p.stem for p in PRESETS_DIR.glob("*.json")))
            elif path.startswith("/api/presets/"):
                name = unquote(path.rsplit("/", 1)[1])
                file = PRESETS_DIR / f"{name}.json"
                if not _PRESET_NAME.match(name) or not file.is_file():
                    return self._error("no such preset", HTTPStatus.NOT_FOUND)
                self._json(merge_defaults(json.loads(file.read_text(encoding="utf-8"))))
            else:
                self._error("not found", HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if not self._same_origin():
                return self._error("cross-origin request refused", HTTPStatus.FORBIDDEN)
            path = self.path.split("?", 1)[0]
            try:
                data = self._body()
                if path == "/api/run":
                    app.runner.start(data.get("config") or {})
                    self._json({"ok": True})
                elif path == "/api/stop":
                    app.runner.stop()
                    self._json({"ok": True})
                elif path == "/api/catalog/refresh":
                    if not app.catalog_lock.acquire(blocking=False):
                        return self._error("the catalog is already being fetched")
                    try:
                        self._json(refresh_catalog())
                    finally:
                        app.catalog_lock.release()
                elif path == "/api/presets":
                    name = str(data.get("name", "")).strip()
                    if not _PRESET_NAME.match(name):
                        return self._error("name: letters, digits, space, - and _ (max 64)")
                    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
                    (PRESETS_DIR / f"{name}.json").write_text(
                        json.dumps(merge_defaults(data.get("config") or {}), indent=1, ensure_ascii=False),
                        encoding="utf-8")
                    self._json({"ok": True})
                else:
                    self._error("not found", HTTPStatus.NOT_FOUND)
            except (ValueError, RuntimeError) as exc:
                self._error(str(exc))
            except Exception as exc:  # noqa: BLE001 - report to the panel
                self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(host="127.0.0.1", port=8765):
    app = App()
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return httpd, app
