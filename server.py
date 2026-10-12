"""Local web interface and session-scoped JSON endpoints."""
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import urlsplit

from .agent import Agent
from .clinic import Clinic
from .learning import improve, load_report
from .policy import Policy

WEB = Path(__file__).parent / "web"
LOGGER = logging.getLogger(__name__)
SESSION_TTL = 1800


class SessionExpired(ValueError):
    pass


class DemoServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    request_queue_size = 32
    allow_reuse_address = not hasattr(socket, 'SO_EXCLUSIVEADDRUSE')

    def server_bind(self):
        # Windows otherwise lets another server bind the same localhost port.
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, address, data_dir="runtime", client=None):
        self.gemini_client = client
        self.loop_status = {}
        folder = Path(data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        self.policy_path = folder / "active_policy.json"
        self.policy = Policy.load(self.policy_path)
        self.clinic = Clinic(folder / "clinic.sqlite3")
        self.sessions = {}
        self.sessions_lock = threading.RLock()
        self.loop_lock = threading.Lock()
        self.workers = threading.BoundedSemaphore(24)
        self.output_dir = folder / "evaluation"
        try:
            self.summary = load_report(self.output_dir)
            self.summary["applied_to_new_conversations"] = self.policy.version == self.summary["active_policy"]["version"]
            if (self.summary.get('mode') == 'gemini') != bool(client):
                self.summary = None
        except (OSError, ValueError, KeyError, TypeError):
            self.summary = None
        try:
            super().__init__(address, Handler)
        except Exception:
            self.clinic.close()
            raise

    def process_request(self, request, client_address):
        if not self.workers.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.workers.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.workers.release()

    def shutdown_request(self, request):
        try:
            request.shutdown(socket.SHUT_WR)
            request.settimeout(0.05)
            remaining = 16384
            # A close with unread bytes can reset the connection before an error
            # response reaches a Windows client. Drain a bounded amount first.
            while remaining:
                data = request.recv(min(4096, remaining))
                if not data:
                    break
                remaining -= len(data)
        except OSError:
            pass
        finally:
            self.close_request(request)

    def server_close(self):
        super().server_close()
        self.clinic.close()

    def run_gemini_loop(self, baseline=None, fresh=False):
        from .gemini import GeminiClient, GeminiError
        from .gemini_learning import improve_gemini
        events = []
        def progress(message):
            self.loop_status = {'running': True, 'progress': message, 'events': list(events)}
        def on_event(event):
            events.append(event)
            progress(event['message'])
        try:
            client = GeminiClient(self.gemini_client.settings, cache_path=self.output_dir / 'model-cache.json')
            result = improve_gemini(self.output_dir, client, baseline or self.policy, progress=progress,
                                    proposer_client=client.repair_client(), fresh=fresh, on_event=on_event)
            summary = load_report(self.output_dir)
            summary['applied_to_new_conversations'] = False
            self.summary = summary
            if result['gate']['promoted']:
                policy = Policy.from_dict(result['active_policy'])
                policy.save(self.policy_path)
                with self.sessions_lock:
                    self.policy = policy
            summary['applied_to_new_conversations'] = self.policy.version == summary['active_policy']['version']
            self.summary = summary
            self.loop_status = {}
        except Exception as error:
            self.loop_status = {'running': False, 'loop_error': str(error) if isinstance(error, GeminiError)
                                else 'The Gemini experiment failed. The active policy was not changed.', 'events': list(events)}
            LOGGER.error('Gemini evaluation failed (%s)', type(error).__name__)
        finally:
            self.loop_lock.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(8)
        super().setup()

    def log_message(self, *args):
        pass

    def _send(self, status, content, mime="application/json; charset=utf-8", filename=None):
        data = json.dumps(content, ensure_ascii=False, allow_nan=False).encode() if mime.startswith("application/json") else content
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _local_request(self):
        try:
            host = urlsplit("http://" + self.headers.get("Host", ""))
            if host.netloc not in {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"} or host.path or host.query or host.fragment:
                raise ValueError("Invalid host")
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != "http" or parsed.netloc not in {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"} or parsed.path or parsed.query or parsed.fragment:
                    raise ValueError("Invalid origin")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise ValueError("Cross-site request")
        except ValueError:
            self._send(403, {"error": "Only requests from this local app are allowed"})
            return False
        return True

    def _session(self):
        token = self.headers.get("X-Session-ID", "")
        with self.server.sessions_lock:
            item = self.server.sessions.get(token)
            if not item or time.monotonic() - item["last_used"] > SESSION_TTL:
                if item:
                    del self.server.sessions[token]
                raise SessionExpired("Conversation expired. Start a new conversation; existing bookings remain in the clinic record.")
            item["last_used"] = time.monotonic()
            return item

    def _snapshot(self, item):
        return {**item["agent"].snapshot(), "acknowledged_message_ids": list(item["messages"])}

    def do_GET(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self._send(200, {"status": "ok", "clinic_today": self.server.clinic.today.isoformat(),
                                    "policy": self.server.policy.to_dict(), "synthetic_only": True,
                                    "mode": 'gemini' if self.server.gemini_client else 'offline',
                                    "model": self.server.gemini_client.model if self.server.gemini_client else None})
        if path == "/api/session":
            try:
                item = self._session()
                with item["lock"]:
                    return self._send(200, self._snapshot(item))
            except SessionExpired as error:
                return self._send(401, {"error": str(error)})
        if path == "/api/evaluation":
            # Old scores must not look like results from the experiment in progress.
            summary = {"pending": True} if self.server.loop_status.get('running') else (self.server.summary or {"pending": True})
            return self._send(200, {**summary, **self.server.loop_status})
        if path == "/api/report":
            if not self.server.summary:
                return self._send(404, {"error": "Run an evaluation first"})
            try:
                return self._send(200, (self.server.output_dir / "report.md").read_bytes(), "text/markdown; charset=utf-8", "careloop-report.md")
            except OSError:
                return self._send(404, {"error": "The saved report is unavailable. Run the evaluation again."})
        static = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in static:
            name, mime = static[path]
            return self._send(200, (WEB / name).read_bytes(), mime)
        self._send(404, {"error": "Not found"})

    def do_POST(self):
        if not self._local_request():
            return
        if self.headers.get("Transfer-Encoding"):
            return self._send(400, {"error": "Chunked requests are not supported"})
        if self.headers.get_content_type() != "application/json":
            return self._send(415, {"error": "Use application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 12000:
                return self._send(413, {"error": "Request must be between 1 and 12000 bytes"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
        except (ValueError, UnicodeDecodeError):
            return self._send(400, {"error": "Invalid JSON request"})
        except TimeoutError:
            return self._send(408, {"error": "Request timed out"})
        path = urlsplit(self.path).path
        try:
            if path == "/api/session":
                if type(body.get("baseline", False)) is not bool:
                    raise ValueError("baseline must be boolean")
                with self.server.sessions_lock:
                    expired = [k for k, v in self.server.sessions.items() if time.monotonic() - v["last_used"] > SESSION_TTL]
                    for key in expired:
                        del self.server.sessions[key]
                    if len(self.server.sessions) >= 128:
                        return self._send(429, {"error": "Too many active conversations. Try again after inactive sessions expire."})
                    interpreter = None
                    if self.server.gemini_client:
                        from .language import GeminiInterpreter
                        interpreter = GeminiInterpreter(self.server.gemini_client)
                    agent = Agent(self.server.clinic, Policy() if body.get("baseline") else self.server.policy, interpreter=interpreter)
                    item = {"agent": agent, "lock": threading.RLock(),
                        "last_used": time.monotonic(), "messages": OrderedDict()}
                    self.server.sessions[agent.session_id] = item
                return self._send(201, self._snapshot(item))
            if path == "/api/chat":
                item = self._session()
                message_id = body.get("message_id")
                if message_id is not None and (not isinstance(message_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,80}", message_id)):
                    raise ValueError("Invalid message_id")
                with item["lock"]:
                    if message_id and message_id in item["messages"]:
                        if item["messages"][message_id] != body.get("message"):
                            return self._send(409, {"error": "A message ID cannot be reused with different text"})
                        return self._send(200, self._snapshot(item))
                    expected_turn = body.get("expected_turn")
                    if expected_turn is not None:
                        if type(expected_turn) is not int or expected_turn < 0:
                            raise ValueError("expected_turn must be a nonnegative integer")
                        if expected_turn != item["agent"].turn:
                            return self._send(409, {"error": "This conversation changed. Review the latest details before replying."})
                    item["agent"].reply(body.get("message"))
                    if message_id:
                        item["messages"][message_id] = body["message"]
                        if len(item["messages"]) > 120:
                            item["messages"].popitem(last=False)
                    return self._send(200, self._snapshot(item))
            if path == "/api/improve":
                if type(body.get('baseline', False)) is not bool:
                    raise ValueError('baseline must be boolean')
                if not self.server.loop_lock.acquire(blocking=False):
                    return self._send(409, {"error": "An improvement loop is already running"})
                if self.server.gemini_client:
                    initial = body.get('baseline', False)
                    baseline = Policy() if initial else self.server.policy
                    self.server.loop_status = {'running': True, 'progress': 'Starting initial-policy comparison' if initial else 'Starting Gemini evaluation', 'events': []}
                    try:
                        threading.Thread(target=self.server.run_gemini_loop, args=(baseline, initial), daemon=True).start()
                    except Exception:
                        self.server.loop_lock.release()
                        raise
                    return self._send(202, self.server.loop_status)
                try:
                    result = improve(self.server.output_dir)
                    summary = load_report(self.server.output_dir)
                    summary["applied_to_new_conversations"] = False
                    self.server.summary = summary
                    if result["gate"]["promoted"]:
                        promoted = Policy.from_dict(result["active_policy"])
                        promoted.save(self.server.policy_path)
                        with self.server.sessions_lock:
                            self.server.policy = promoted
                    summary["applied_to_new_conversations"] = self.server.policy.version == summary["active_policy"]["version"]
                    return self._send(200, summary)
                finally:
                    self.server.loop_lock.release()
            return self._send(404, {"error": "Not found"})
        except SessionExpired as error:
            return self._send(401, {"error": str(error)})
        except ValueError as error:
            return self._send(400, {"error": str(error)})
        except Exception as error:
            LOGGER.error("Request failed: %s (%s)", path, type(error).__name__)
            message = "The clinic service failed. Check this conversation before starting another booking; a booking may have committed." if path == "/api/chat" else "The operation failed. Check the latest report or try again."
            return self._send(500, {"error": message})


def serve(port=8000, data_dir="runtime", client=None):
    server = DemoServer(("127.0.0.1", port), data_dir, client)
    print(f"CareLoop: http://127.0.0.1:{server.server_port}", flush=True)
    print("Fictional patients only. Press Ctrl+C to stop.", flush=True)
    print('Language: ' + (client.model if client else 'offline rules'), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
