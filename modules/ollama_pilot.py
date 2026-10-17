"""Local Ollama Pilot: a PilotAdapter that chooses ONE legal affordance by id. Local inference only.

What it can do: read the canonical controller context it is handed, ask the local model to pick an affordance_id, validate the answer, and
return it as an ordinary AgentBrainResponse. Everything after that (grounding, ActionGate, translation, execution) is the same Middleware
path the Procedural Baseline uses.

What it cannot do: touch the Environment, the filesystem, a shell or any tool; create projects or targets; supply targets or arguments the
chosen affordance does not already carry; fall back to another controller. When parsing/validation keeps failing after the configured
bounded retries it raises ControllerFailure; it never returns a substituted action.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from modules.action_schema import ExecutableActionType
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse
from modules.controller_context import assert_canonical_context_clean
from modules.ollama_client import OllamaClient, OllamaError, OllamaModelMissingError, OllamaUnavailableError
from modules.ollama_prompt import build_prompt, estimate_tokens

IMPLEMENTATION_VERSION = "1"

# ---- seam for tests and tooling: how a client is made. Production: the real loopback client. Nothing else may create network clients.
client_factory: Callable[[str, float], Any] = lambda base_url, timeout_s: OllamaClient(base_url=base_url, timeout_s=timeout_s)


class ControllerFailure(RuntimeError):
    """The Ollama Pilot could not produce a valid legal choice. Carries the full decision trace."""

    def __init__(self, message: str, trace: Dict[str, Any]):
        super().__init__(message)
        self.trace = trace


class ControllerUnavailableError(RuntimeError):
    """Preflight: the local Ollama service or the selected model is not usable. Raised before any simulation starts."""


def preflight(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Check service + installed model for one Ollama controller. Raises ControllerUnavailableError with a clear reason. Never pulls a model."""
    model = str(settings.get("model") or "").strip()
    if not model:
        raise ControllerUnavailableError("local_ollama controller has no model selected")
    client = client_factory(settings.get("base_url"), float(settings.get("timeout_s") or 120.0))
    status = client.status()
    if not status.get("available"):
        raise ControllerUnavailableError(f"Local Ollama is not running (no response at {status.get('base_url')}).\nOpen the Ollama app, wait for it to start, then click \"Check Ollama / refresh models\".\n(Nothing is started or downloaded automatically.) Detail: {status.get('error')}")
    try:
        installed = client.installed_model_names()
    except OllamaError as exc:
        raise ControllerUnavailableError(f"could not list local Ollama models: {exc}") from exc
    resolved = OllamaClient.model_matches(model, installed)
    if resolved is None:
        raise ControllerUnavailableError(f"model {model!r} is not installed locally (installed: {installed or 'none'}). Models are never pulled automatically; run `ollama pull` yourself.")
    return {"available": True, "version": status.get("version"), "resolved_model": resolved, "installed_models": installed, "base_url": status.get("base_url")}


def warm_up(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Load the selected (already installed) model before simulated time starts. -> {warmed, latency_s, load_duration_s} or {warmed: False, error}.
    Failure is reported, not fatal: the first real call is then simply cold, and any real problem surfaces through the normal inference error path."""
    client = client_factory(settings.get("base_url"), float(settings.get("timeout_s") or 120.0))
    if not hasattr(client, "warm"):
        return {"warmed": False, "error": "client has no warm-up support"}
    try:
        info = client.warm(model=str(settings["model"]), keep_alive=str(settings.get("keep_alive") or "30m"), timeout_s=max(300.0, float(settings.get("timeout_s") or 120.0)))
    except OllamaError as exc:
        return {"warmed": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"warmed": True, **info}


def _parse_json_object(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, flags=re.S)
    if fence:
        t = fence.group(1)
    value = json.loads(t)
    if not isinstance(value, dict):
        raise ValueError("reply is not a JSON object")
    return value


class OllamaPilotAdapter:
    pilot_id = "local_ollama"
    display_name = "Local Ollama Pilot"
    # Agent-side wiring flags (the Mecha reads these; it has no knowledge of Ollama itself).
    synchronous_llm_pilot = True  # decisions are taken synchronously in the Pilot path; no async planner, DIK-integration or bootstrap fallback
    consumes_canonical_context_only = True

    def __init__(self, settings: Dict[str, Any], *, agent_id: str, role: str, estimator=None):
        self.settings = dict(settings)
        self.estimator = estimator  # session-adaptive TokenEstimator shared per model tag (None = fixed conservative 2.8 chars/token)
        self.agent_id = agent_id
        self.role = role
        self.model = str(settings["model"])
        self.client = client_factory(settings.get("base_url"), float(settings.get("timeout_s") or 120.0))
        self.last_trace: Optional[Dict[str, Any]] = None
        self.traces_written = 0
        self.runtime_status = "ready"  # "ready" | "thinking" | "error" (per-agent GUI label); written by whoever drives the call
        self.cancelled = False  # set when the run stops: no further attempts are started and the late result is discarded

    def cancel(self) -> None:
        self.cancelled = True

    # ------------------------------------------------------------------ PilotAdapter protocol
    def choose_fallback_action(self, *, agent=None, context_packet=None, reason="", sim_state=None):
        return None  # no cross-controller or silent fallback, ever

    def handle_blocked_action(self, *, agent=None, original_decision=None, gate_result=None, sim_state=None):
        return None  # the model sees the blocker in recent outcomes on its next turn and chooses again

    # ------------------------------------------------------------------ budgets / options
    def _budget_tokens(self) -> int:
        s = self.settings
        if s.get("prompt_budget_tokens"):
            return int(s["prompt_budget_tokens"])
        return max(512, int(s.get("max_context_tokens") or 8192) - int(s.get("max_output_tokens") or 400) - 256)

    def _options(self) -> Dict[str, Any]:
        s = self.settings
        opts: Dict[str, Any] = {"temperature": float(s.get("temperature") if s.get("temperature") is not None else 0.2),
                                "num_predict": int(s.get("max_output_tokens") or 400), "num_ctx": int(s.get("max_context_tokens") or 8192)}
        if s.get("controller_generation_seed") is not None:
            opts["seed"] = int(s["controller_generation_seed"])
        for k, v in dict(s.get("inference_parameters") or {}).items():
            opts.setdefault(k, v)
        return opts

    # ------------------------------------------------------------------ decision
    def choose_action(self, request: AgentBrainRequest) -> AgentBrainResponse:
        canonical = assert_canonical_context_clean(dict((request.task_context or {}).get("canonical_context") or {}))
        if not canonical:
            raise ControllerFailure("no canonical context in request", {"error": "missing_canonical_context"})
        built = build_prompt(canonical, budget_tokens=self._budget_tokens(), prompt_version=str(self.settings.get("prompt_version") or "canonical_choice_v2"), estimator=self.estimator)
        amap, schema = built["affordance_map"], built["schema"]
        trace: Dict[str, Any] = {
            "kind": "ollama_decision", "request_id": request.request_id, "agent_id": self.agent_id, "role": self.role, "sim_time": float(request.sim_time),
            "model": self.model, "options": self._options(), "prompt_version": built["metadata"]["prompt_version"], "prompt_hash": built["prompt_hash"],
            "system_prompt": built["system"], "user_prompt": built["user"], "budget": built["metadata"], "attempts": [], "validation_outcome": None,
            "retry_count": 0, "parsed_affordance_id": None, "parsed_action": None, "wall_clock_inference_s": 0.0,
        }
        self.last_trace = trace
        if not amap:
            trace["validation_outcome"] = "no_legal_affordances"
            raise ControllerFailure("no legal affordances were offered", trace)
        messages: List[Dict[str, str]] = [{"role": "system", "content": built["system"]}, {"role": "user", "content": built["user"]}]
        max_retries = int(self.settings.get("max_retries", 2))
        for attempt in range(max_retries + 1):
            if self.cancelled:
                trace["validation_outcome"] = "cancelled"
                raise ControllerFailure("run stopped while inference was pending", trace)
            record: Dict[str, Any] = {"attempt": attempt, "raw_response": None, "error": None, "latency_s": None, "token_metadata": {}, "done_reason": None}
            trace["attempts"].append(record)
            trace["retry_count"] = attempt
            try:
                reply = self.client.chat(model=self.model, messages=messages, options=self._options(), response_format=schema, think=bool(self.settings.get("think", False)),
                                         timeout_s=float(self.settings.get("timeout_s") or 120.0), **({"keep_alive": str(self.settings["keep_alive"])} if self.settings.get("keep_alive") else {}))
            except OllamaError as exc:  # transport/timeout/service problems: recorded, retried within the same bound, never replaced by another controller
                record["error"] = f"{type(exc).__name__}: {exc}"
                trace["validation_outcome"] = "inference_error"
                correction = None
            else:
                record.update(raw_response=reply["content"], latency_s=reply["latency_s"], token_metadata=reply["token_metadata"], done_reason=reply.get("done_reason"))
                used = (reply["token_metadata"] or {}).get("prompt_eval_count")
                if attempt == 0 and used is not None:  # first attempt only: later attempts carry extra correction messages
                    obs = self.estimator.observe(built["metadata"]["prompt_chars"], int(used), built["metadata"]["estimated_prompt_tokens"]) if self.estimator is not None else None
                    trace["token_estimation"] = obs or {"chars": built["metadata"]["prompt_chars"], "actual_prompt_tokens": int(used), "estimated_prompt_tokens": built["metadata"]["estimated_prompt_tokens"],
                                                        "estimate_over_actual": round(built["metadata"]["estimated_prompt_tokens"] / max(1, int(used)), 4)}
                ctx_limit = int(self.settings.get("max_context_tokens") or 8192)
                if used is not None and int(used) >= int(0.97 * ctx_limit):
                    record["context_window_nearly_full"] = True  # Ollama may truncate beyond num_ctx; raise max_context_tokens or lower the prompt budget
                    trace["context_window_nearly_full"] = True
                trace["wall_clock_inference_s"] += float(reply["latency_s"])
                parsed, error = self._validate(reply["content"], amap)
                if parsed is not None:
                    aff_id, args, note = parsed
                    aff = amap[aff_id]
                    trace.update(validation_outcome="valid", parsed_affordance_id=aff_id, parsed_action={"action_type": aff.get("action_type"), "target_id": aff.get("target_id"), "action_args": dict(aff.get("action_args") or {})}, decision_note=note)
                    return self._response(request, aff, note, trace)
                record["error"] = error
                trace["validation_outcome"] = f"rejected:{error.split(':')[0]}"
                correction = self._correction(error, amap)
            if attempt < max_retries:
                if record.get("raw_response") is not None:
                    messages = messages + [{"role": "assistant", "content": record["raw_response"]}, {"role": "user", "content": correction}]
                time.sleep(0)  # no backoff: a local service; the bound is the retry count
        trace["validation_outcome"] = trace["validation_outcome"] or "exhausted"
        raise ControllerFailure(f"Ollama controller for {self.role} produced no valid choice after {max_retries + 1} attempt(s): {trace['attempts'][-1].get('error')}", trace)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _validate(raw: str, amap: Dict[str, Dict[str, Any]]):
        """-> ((affordance_id, arguments, note), None) or (None, error). The model can only pick an offered affordance and cannot add arguments."""
        try:
            obj = _parse_json_object(raw)
        except ValueError as exc:
            return None, f"malformed_json: {exc}"
        aff_id = obj.get("affordance_id")
        if not isinstance(aff_id, str) or aff_id not in amap:
            return None, f"unknown_affordance: {aff_id!r} is not one of {sorted(amap)[:40]}"
        args = obj.get("arguments") or {}
        if not isinstance(args, dict):
            return None, "arguments_not_an_object"
        permitted = dict(amap[aff_id].get("action_args") or {})
        for k, v in args.items():
            if k not in permitted or permitted[k] != v:
                return None, f"arguments_not_permitted: {k!r} is not an argument of {aff_id}"
        note = obj.get("decision_note")
        if note is not None and not isinstance(note, str):
            return None, "decision_note_not_a_string"
        return (aff_id, permitted, (note or "")[:200]), None

    @staticmethod
    def _correction(error: str, amap: Dict[str, Dict[str, Any]]) -> str:
        legal = "; ".join(f"{i}={a.get('action_type')}:{a.get('target_id')}" for i, a in list(amap.items())[:60])
        return (f"Your reply was rejected ({error}). Reply with only a JSON object {{\"affordance_id\": ..., \"arguments\": {{}}, \"decision_note\": ...}} "
                f"where affordance_id is exactly one of the currently executable ids: {legal}.")

    def _response(self, request: AgentBrainRequest, aff: Dict[str, Any], note: str, trace: Dict[str, Any]) -> AgentBrainResponse:
        step = {"step_index": 0, "action_type": ExecutableActionType(aff["action_type"]).value, "target_id": aff.get("target_id"), "target_zone": aff.get("target_zone"),
                "expected_purpose": note or f"chose {trace['parsed_affordance_id']}", "action_args": dict(aff.get("action_args") or {})}
        return AgentBrainResponse.from_dict({
            "response_id": f"ollama-{request.request_id}",
            "agent_id": request.agent_id,
            "plan": {"plan_id": f"ollama-plan-{request.request_id}", "plan_horizon": 1,
                     "ordered_goals": [{"goal_id": "local_ollama_choice", "description": "chosen affordance", "priority": 0.5, "status": "active", "source": "planner"}],
                     "ordered_actions": [step], "next_action": step, "confidence": 0.5,
                     "notes": [f"controller=local_ollama model={self.model} affordance={trace['parsed_affordance_id']} prompt_hash={trace['prompt_hash']}"]},
            "explanation": note or None,
        })


class OllamaControllerProvider:
    """Runtime-shape shim so per-agent runtime bookkeeping (provider class name, last_outcome) keeps working. It performs no inference itself."""

    def __init__(self, adapter: OllamaPilotAdapter):
        self.adapter = adapter
        self.last_outcome: Optional[Dict[str, Any]] = None

    def generate_plan(self, request_packet):  # pragma: no cover - the adapter is the only caller path
        return self.adapter.choose_action(request_packet)
