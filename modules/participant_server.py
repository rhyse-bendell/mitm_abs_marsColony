"""Local participant server: serves the browser client and a WebSocket event channel. Standard library only.

Why stdlib (documented choice): the repository has no web/WebSocket dependency, the brief asks for no heavyweight framework, and dependencies must not be
changed to suit a feature. A minimal RFC 6455 text-frame implementation (~100 lines) is enough for one JSON message stream per participant.

Safety defaults
* Binds 127.0.0.1 unless the host is passed explicitly (`bind_host="0.0.0.0"` is a deliberate, later, LAN decision; it is not exposed in the first slice's UI).
* A random per-session token is required on the page URL and the WebSocket URL; a cross-origin browser page cannot open the socket (Origin must match Host).
* One client per participant: a second connection for the same role is refused until the first drops.
* The server thread only touches ParticipantGateway (queues + the last published snapshot). It never reads or writes simulation objects.
* Closing the tab, killing the browser or a 20 s silence marks the participant disconnected; the agent then idles safely (no intents arrive, nothing is faked).
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CLIENT_DIR = Path(__file__).resolve().parent / "participant_client"
MAX_FRAME = 64 * 1024
SILENCE_LIMIT_S = 20.0
_CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


# ---- minimal WebSocket framing ---------------------------------------------------------------------------------------------------
def encode_frame(text: str, opcode: int = 0x1) -> bytes:
    payload = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack(">H", n)
    else:
        head += bytes([127]) + struct.pack(">Q", n)
    return head + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def read_frame(sock: socket.socket):
    """-> (opcode, payload bytes). Raises socket.timeout when idle, ConnectionError on close, ValueError on an unsupported/oversized frame."""
    b1, b2 = _recv_exact(sock, 2)
    fin, opcode, masked, n = b1 & 0x80, b1 & 0x0F, b2 & 0x80, b2 & 0x7F
    if n == 126:
        n = struct.unpack(">H", _recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
    if n > MAX_FRAME or not fin or opcode == 0x0:
        raise ValueError("unsupported frame")
    if not masked:
        raise ValueError("client frames must be masked")
    mask = _recv_exact(sock, 4)
    data = bytearray(_recv_exact(sock, n))
    for i in range(n):
        data[i] ^= mask[i % 4]
    return opcode, bytes(data)


class ParticipantServer:
    """Serves one gateway per role. `url_for(role)` gives the participant URL (with token)."""

    def __init__(self, gateways: Dict[str, "ParticipantGateway"], *, bind_host: str = "127.0.0.1", port: int = 0, token: Optional[str] = None, push_interval_s: float = 0.1):
        self.gateways = dict(gateways)
        self.token = token or secrets.token_urlsafe(12)
        self.bind_host = bind_host
        self.push_interval_s = push_interval_s
        self._stop = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                pass

            def _deny(self, code: int, why: str):
                body = why.encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                q = parse_qs(url.query)
                if url.path == "/ws":
                    return server._handle_ws(self, q)
                if q.get("token", [""])[0] != server.token and not url.path.startswith("/static/"):
                    return self._deny(403, "forbidden")
                if url.path in ("/", "/index.html"):
                    return self._file("index.html")
                if url.path.startswith("/static/"):
                    return self._file(url.path.split("/static/", 1)[1])
                return self._deny(404, "not found")

            def _file(self, name: str):
                path = (CLIENT_DIR / name).resolve()
                if CLIENT_DIR.resolve() not in path.parents or not path.is_file():
                    return self._deny(404, "not found")
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((bind_host, port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------------------------------------------------------
    def start(self) -> "ParticipantServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="participant-server", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass
        for gw in self.gateways.values():
            gw.on_disconnect()

    def url_for(self, role: str) -> str:
        host = "127.0.0.1" if self.bind_host in {"0.0.0.0", ""} else self.bind_host
        return f"http://{host}:{self.port}/?role={role}&token={self.token}"

    # ---- websocket ---------------------------------------------------------------------------------------------------------
    def _handle_ws(self, h: BaseHTTPRequestHandler, q: Dict[str, list]) -> None:
        role = q.get("role", [""])[0]
        gw = self.gateways.get(role)
        if q.get("token", [""])[0] != self.token or gw is None:
            return h._deny(403, "forbidden")
        origin, host = h.headers.get("Origin"), h.headers.get("Host", "")
        if origin and urlparse(origin).netloc != host:
            return h._deny(403, "cross-origin websocket refused")
        key = h.headers.get("Sec-WebSocket-Key")
        if (h.headers.get("Upgrade", "").lower() != "websocket") or not key:
            return h._deny(400, "websocket upgrade required")
        if not gw.on_connect():
            return h._deny(409, "a participant client is already connected for this role")
        accept = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        h.send_response(101, "Switching Protocols")
        h.send_header("Upgrade", "websocket")
        h.send_header("Connection", "Upgrade")
        h.send_header("Sec-WebSocket-Accept", accept)
        h.end_headers()
        h.wfile.flush()
        sock = h.connection
        sock.settimeout(self.push_interval_s)
        last_heard = time.monotonic()
        last_seq = 0
        try:
            sock.sendall(encode_frame(gw.layout_json))
            while not self._stop.is_set():
                try:
                    opcode, payload = read_frame(sock)
                    last_heard = time.monotonic()
                    if opcode == 0x8:
                        break
                    if opcode == 0x9:
                        sock.sendall(encode_frame(payload.decode("latin-1"), 0xA))
                    elif opcode == 0x1:
                        try:
                            msg = json.loads(payload.decode("utf-8"))
                        except ValueError:
                            msg = None
                        reply = gw.on_message(msg) if isinstance(msg, dict) else {"t": "ack", "ok": False, "reason": "bad_json"}
                        if reply:
                            sock.sendall(encode_frame(json.dumps(reply)))
                except socket.timeout:
                    pass
                if time.monotonic() - last_heard > SILENCE_LIMIT_S:
                    break
                seq, text = gw.latest()
                if seq != last_seq and text:
                    sock.sendall(encode_frame(text))
                    last_seq = seq
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            gw.on_disconnect()
            try:
                sock.sendall(encode_frame("", 0x8))
            except OSError:
                pass
            h.close_connection = True
