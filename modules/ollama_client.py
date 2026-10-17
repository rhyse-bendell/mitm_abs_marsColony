"""Minimal client for the LOCAL Ollama service (loopback only).

Hard constraints enforced here, in code:
- the endpoint host must be a loopback address (127.0.0.0/8, ::1, or the literal name "localhost"); anything else is rejected before any socket is opened;
- plain http only; environment proxies are bypassed; redirects are refused;
- the only endpoints ever used are GET /api/version, GET /api/tags and POST /api/chat. There is NO pull/download/delete/create call,
  no automatic start of the service, no API key and no authentication header.
"""
from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
ALLOWED_PATHS = ("/api/version", "/api/tags", "/api/chat", "/api/ps")


class OllamaError(RuntimeError):
    """Base class for local-Ollama problems."""


class NonLoopbackEndpointError(OllamaError):
    """The configured endpoint is not a loopback address."""


class OllamaUnavailableError(OllamaError):
    """The local Ollama service did not answer."""


class OllamaModelMissingError(OllamaError):
    """The selected model is not installed locally (it is never pulled automatically)."""


class OllamaRequestError(OllamaError):
    """A chat request failed after the service was reached (timeout, HTTP error, malformed reply)."""


def validate_loopback_endpoint(base_url: str) -> str:
    """Return the normalised base URL, or raise NonLoopbackEndpointError. No DNS lookup is performed."""
    parsed = urllib.parse.urlparse(str(base_url or "").strip())
    if parsed.scheme != "http":
        raise NonLoopbackEndpointError(f"only plain http to a loopback host is allowed, got scheme {parsed.scheme!r} in {base_url!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise NonLoopbackEndpointError(f"endpoint has no host: {base_url!r}")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise NonLoopbackEndpointError(f"endpoint host {host!r} is not a loopback address")
        except ValueError:
            raise NonLoopbackEndpointError(f"endpoint host {host!r} is not a loopback address or the literal 'localhost'") from None
    if parsed.username or parsed.password:
        raise NonLoopbackEndpointError("credentials in the endpoint are not allowed")
    netloc = parsed.netloc
    return f"http://{netloc}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401
        raise OllamaRequestError("redirects are refused for the local Ollama client")


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 120.0):
        self.base_url = validate_loopback_endpoint(base_url)
        self.timeout_s = float(timeout_s)
        # ProxyHandler({}) ignores HTTP(S)_PROXY from the environment so a loopback request can never be routed off-machine.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    # ------------------------------------------------------------------ transport
    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        if path not in ALLOWED_PATHS:
            raise OllamaRequestError(f"path {path!r} is not an allowed local-Ollama endpoint")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=float(timeout_s or self.timeout_s)) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise OllamaRequestError(f"HTTP {exc.code} from local Ollama {path}: {detail}") from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if "timed out" in str(reason).lower():
                raise OllamaRequestError(f"timeout after {timeout_s or self.timeout_s}s calling local Ollama {path}") from exc
            raise OllamaUnavailableError(f"local Ollama not reachable at {self.base_url}: {reason}") from exc
        try:
            return json.loads(body)
        except ValueError as exc:
            raise OllamaRequestError(f"non-JSON reply from local Ollama {path}: {body[:200]!r}") from exc

    # ------------------------------------------------------------------ status / models
    def status(self) -> Dict[str, Any]:
        try:
            version = self._request("GET", "/api/version", timeout_s=min(5.0, self.timeout_s))
            return {"available": True, "base_url": self.base_url, "version": version.get("version"), "error": None}
        except OllamaError as exc:
            return {"available": False, "base_url": self.base_url, "version": None, "error": str(exc)}

    def list_models(self) -> List[Dict[str, Any]]:
        reply = self._request("GET", "/api/tags", timeout_s=min(10.0, self.timeout_s))
        out = []
        for m in reply.get("models") or []:
            details = m.get("details") or {}
            out.append({"name": m.get("name") or m.get("model"), "parameter_size": details.get("parameter_size"), "quantization": details.get("quantization_level"),
                        "context_length": details.get("context_length"), "family": details.get("family"), "size_bytes": m.get("size"), "digest": (m.get("digest") or "")[:12]})
        return sorted(out, key=lambda r: str(r["name"]))

    def installed_model_names(self) -> List[str]:
        return [str(m["name"]) for m in self.list_models()]

    def loaded_models(self) -> List[Dict[str, Any]]:
        """Models currently resident in memory (GET /api/ps, read-only). Used only for diagnostics."""
        reply = self._request("GET", "/api/ps", timeout_s=min(3.0, self.timeout_s))
        return [{"name": m.get("name") or m.get("model"), "size_bytes": m.get("size"), "size_vram_bytes": m.get("size_vram")} for m in reply.get("models") or []]

    @staticmethod
    def model_matches(requested: str, installed: List[str]) -> Optional[str]:
        """Exact tag, or an untagged request matching `<name>:latest`. Never a fuzzy match."""
        if requested in installed:
            return requested
        if ":" not in requested and f"{requested}:latest" in installed:
            return f"{requested}:latest"
        return None

    # ------------------------------------------------------------------ inference
    def warm(self, *, model: str, keep_alive: str = "30m", timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Load an ALREADY-INSTALLED model into memory (POST /api/chat with an empty message list; Ollama's documented load-only call) and keep it
        resident for `keep_alive`. Generates nothing. A missing model is an error here, never a download."""
        started = time.perf_counter()
        reply = self._request("POST", "/api/chat", {"model": model, "messages": [], "stream": False, "keep_alive": keep_alive}, timeout_s=timeout_s)
        return {"latency_s": time.perf_counter() - started, "load_duration_s": (reply.get("load_duration") or 0) / 1e9, "done_reason": reply.get("done_reason")}

    def chat(self, *, model: str, messages: List[Dict[str, str]], options: Dict[str, Any], response_format: Any = None, think: Optional[bool] = False,
             timeout_s: Optional[float] = None, keep_alive: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False, "options": dict(options)}
        if keep_alive:
            payload["keep_alive"] = keep_alive
        if response_format is not None:
            payload["format"] = response_format
        if think is not None:
            payload["think"] = bool(think)
        started = time.perf_counter()
        reply = self._request("POST", "/api/chat", payload, timeout_s=timeout_s)
        latency = time.perf_counter() - started
        message = reply.get("message") or {}
        return {
            "content": str(message.get("content") or ""),
            "latency_s": latency,
            "done_reason": reply.get("done_reason"),
            "token_metadata": {k: reply.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration") if k in reply},
        }
