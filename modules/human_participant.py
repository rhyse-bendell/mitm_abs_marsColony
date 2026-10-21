"""Human Participant controller: a person drives ONE agent through the same Pilot -> Mecha -> Environment boundary every controller uses.

What it is
    A per-agent controller (`controller_type = "human_participant"`) with an inbound, thread-safe *intent queue*. A participant client (a browser,
    see participant_server.py) submits intents; the agent's Mecha drains them on the simulation thread, turns each into a canonical typed
    `BrainDecision`, and runs it through the unchanged ActionGate and translation. The Environment decides legality and state changes.

What it is not
    * It never invents, repeats, completes or "improves" an action. An empty queue means the agent simply does nothing (it is not a WAIT choice).
    * It never touches Environment/Construction state and holds no reference to them. It is handed nothing but queues.
    * A blocked/rejected intent is reported back to the participant and logged; nothing is substituted in its place.
    * It never claims to know what the person remembers or believes. DIK traces record *exposure* (what was opened or received), nothing more.

Thread model
    `submit_intent` / `push_feedback` are called from the participant-server thread; `drain` / `feedback_since` from the simulation thread. All shared state
    sits behind one lock. Closing or losing the participant client only flips `connected`; the agent then idles safely (intents stop arriving).
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from modules.action_schema import BrainDecision, ExecutableActionType

CONTROLLER_TYPE = "human_participant"
IMPLEMENTATION_VERSION = "1"
MAX_PENDING_INTENTS = 16
MAX_FEEDBACK = 200

# Actions a participant may submit. Everything else (reassess/observe/wait/meeting/request_assistance, and plan externalization, whose content the
# Mecha would have to author on the person's behalf) is not a human verb in this first slice.
HUMAN_SUBMITTABLE_ACTIONS = frozenset({
    ExecutableActionType.MOVE_TO_TARGET.value,
    ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
    ExecutableActionType.CONSULT_TEAM_ARTIFACT.value,
    ExecutableActionType.COMMUNICATE.value,
    ExecutableActionType.TRANSPORT_RESOURCES.value,
    ExecutableActionType.START_CONSTRUCTION.value,
    ExecutableActionType.CONTINUE_CONSTRUCTION.value,
    ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value,
    ExecutableActionType.VALIDATE_CONSTRUCTION.value,
    ExecutableActionType.CONNECT_STRUCTURES.value,
})

# Coarse scientific category of each canonical action, for research traces (NOT a second action vocabulary).
ACTION_CATEGORY = {
    ExecutableActionType.MOVE_TO_TARGET.value: "movement",
    ExecutableActionType.INSPECT_INFORMATION_SOURCE.value: "information_source",
    ExecutableActionType.CONSULT_TEAM_ARTIFACT.value: "artifact_interaction",
    ExecutableActionType.EXTERNALIZE_PLAN.value: "artifact_interaction",
    ExecutableActionType.COMMUNICATE.value: "chat",
    ExecutableActionType.TRANSPORT_RESOURCES.value: "resource_transport",
    ExecutableActionType.START_CONSTRUCTION.value: "project_start",
    ExecutableActionType.CONTINUE_CONSTRUCTION.value: "construction_contribution",
    ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value: "repair",
    ExecutableActionType.VALIDATE_CONSTRUCTION.value: "validation",
    ExecutableActionType.CONNECT_STRUCTURES.value: "connect_structures",
}

# What the participant is told. Class A (participant-visible) blockers get a specific message. Everything else is class B (internal/hidden-truth
# adjacent: readiness, knowledge, correctness) and collapses to a generic sentence so the UI cannot leak what only the Middleware knows.
PARTICIPANT_BLOCKER_MESSAGES: Dict[str, str] = {
    "private_source_inaccessible": "You are not permitted to view another role's private briefing.",
    "not_in_interaction_range": "Move closer to interact.",
    "busy_committed": "You are busy with another action.",
    "no_pickup_resource_source": "There are no resources left to pick up.",
    "project_does_not_need_material": "That project already has all the materials it needs.",
    "project_already_complete": "That project is already complete.",
    "physical_build_not_started": "Nothing has been built there yet.",
    "physical_incomplete": "The structure is not finished being built yet.",
    "destination_not_navigable": "You can't walk there.",
    "destination_unreachable": "There is no route to that spot.",
    "target_locked_until_bridge_access": "That area is not accessible yet.",
    "unknown_recipient": "That teammate isn't available to receive the message.",
    "empty_message": "Type a message first.",
    "message_too_long": "That message is too long.",
    "stale_option": "That option is no longer available.",
    "not_a_participant_action": "That is not something you can do.",
    "participant_not_connected": "Your participant client is not connected.",
    "experiment_paused": "The experiment is paused.",
    "experiment_ended": "The experiment has ended.",
}
GENERIC_BLOCKER_MESSAGE = "That isn't possible right now."


def participant_message_for(blockers: List[str]) -> str:
    """The single participant-appropriate sentence for a set of raw blockers (first class-A match wins; unknown -> generic)."""
    for blocker in blockers or []:
        key = str(blocker).split(":", 1)[0]
        if key in PARTICIPANT_BLOCKER_MESSAGES:
            return PARTICIPANT_BLOCKER_MESSAGES[key]
    return GENERIC_BLOCKER_MESSAGE


@dataclass
class HumanIntent:
    """One participant request, already reduced to a canonical typed action plus the UI context it came from (never raw keystrokes)."""

    action: str
    target_id: Optional[str] = None
    action_args: Dict[str, Any] = field(default_factory=dict)
    source: str = "menu"  # "menu" | "map_click" | "chat"
    ui_object: Optional[str] = None  # the object the participant had selected (a label for traces; carries no authority)
    label: str = ""
    intent_id: str = ""
    wall_submitted: float = 0.0
    clicked_at: Optional[List[float]] = None


class HumanParticipantController:
    """Inbound intent queue + outbound participant feedback for one human-controlled agent. Holds no world state."""

    controller_type = CONTROLLER_TYPE

    def __init__(self, agent_id: str, role: str, name: str, participant_id: Optional[str] = None, clock=time.time):
        self.agent_id = agent_id
        self.role = role
        self.name = name
        self.participant_id = participant_id or f"participant:{role}"
        self._clock = clock
        self._lock = threading.Lock()
        self._intents: Deque[HumanIntent] = deque()
        self._feedback: Deque[Dict[str, Any]] = deque(maxlen=MAX_FEEDBACK)
        self._ui_events: Deque[Dict[str, Any]] = deque(maxlen=64)
        self._ids = itertools.count(1)
        self._feedback_ids = itertools.count(1)
        self.connected = False
        self.ever_connected = False
        self.last_seen_wall: Optional[float] = None
        self.dropped_intents = 0
        self.closed = False  # the run ended; nothing more is accepted

    # ---- participant-server thread -------------------------------------------------------------------------------------------
    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self.connected = bool(connected)
            self.ever_connected = self.ever_connected or bool(connected)
            self.last_seen_wall = self._clock()

    def touch(self) -> None:
        with self._lock:
            self.last_seen_wall = self._clock()

    def submit_intent(self, intent: HumanIntent) -> Dict[str, Any]:
        """Queue an intent for the simulation thread. -> {"accepted": bool, "intent_id", "reason"}. Acceptance here means *queued*, not legal."""
        with self._lock:
            if self.closed:
                return {"accepted": False, "intent_id": None, "reason": "experiment_ended"}
            if not self.connected:
                return {"accepted": False, "intent_id": None, "reason": "participant_not_connected"}
            if len(self._intents) >= MAX_PENDING_INTENTS:
                self.dropped_intents += 1
                return {"accepted": False, "intent_id": None, "reason": "too_many_pending_intents"}
            intent.intent_id = f"{self.agent_id}#i{next(self._ids)}"
            intent.wall_submitted = self._clock()
            self._intents.append(intent)
            return {"accepted": True, "intent_id": intent.intent_id, "reason": None}

    def submit_ui_event(self, event: Dict[str, Any]) -> None:
        """A non-action interface event worth a research record (menu opened on an object, source panel closed/reopened). Processed on the simulation thread."""
        with self._lock:
            if not self.closed:
                self._ui_events.append(dict(event))

    def feedback_since(self, last_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(f) for f in self._feedback if f["id"] > last_id]

    # ---- simulation thread ---------------------------------------------------------------------------------------------------
    def drain(self) -> List[HumanIntent]:
        with self._lock:
            items = list(self._intents)
            self._intents.clear()
            return items

    def drain_ui_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._ui_events)
            self._ui_events.clear()
            return items

    def push_feedback(self, level: str, message: str, *, intent_id: Optional[str] = None, sim_time: Optional[float] = None, kind: str = "action") -> Dict[str, Any]:
        with self._lock:
            row = {"id": next(self._feedback_ids), "level": level, "kind": kind, "message": message, "intent_id": intent_id,
                   "sim_time": None if sim_time is None else round(float(sim_time), 2), "wall_ts": self._clock()}
            self._feedback.append(row)
            return row

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._intents.clear()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"participant_id": self.participant_id, "connected": self.connected, "ever_connected": self.ever_connected,
                    "pending_intents": len(self._intents), "dropped_intents": self.dropped_intents, "last_seen_wall": self.last_seen_wall}


def intent_to_decision(intent: HumanIntent) -> BrainDecision:
    """A participant intent as the canonical typed decision the Mecha already understands. Arguments are carried verbatim (the gate validates them)."""
    return BrainDecision(
        selected_action=ExecutableActionType(intent.action),
        target_id=intent.target_id,
        action_args=dict(intent.action_args or {}),
        reason_summary=f"human participant intent ({intent.source}): {intent.label or intent.action}",
        confidence=1.0,
    )


class HumanParticipantPilotAdapter:
    """PilotAdapter-shaped runtime object for a human. It never chooses an action: decisions are intents read from the controller queue."""

    pilot_id = CONTROLLER_TYPE
    display_name = "Human Participant"
    human_participant = True  # Agent-side wiring flag (the Mecha reads this; it knows nothing else about humans)
    synchronous_llm_pilot = False
    consumes_canonical_context_only = True

    def __init__(self, controller: HumanParticipantController):
        self.controller = controller
        self.runtime_status = "awaiting_participant"

    def choose_action(self, request):  # pragma: no cover - a human is never asked to choose through the brain request path
        raise RuntimeError("a human participant does not decide through AgentBrainRequest; intents arrive on the controller queue")

    def choose_fallback_action(self, *, agent=None, context_packet=None, reason="", sim_state=None):
        return None  # no fallback controller, ever

    def handle_blocked_action(self, *, agent=None, original_decision=None, gate_result=None, sim_state=None):
        return None  # a blocked intent is reported to the participant, never rerouted


class HumanParticipantProvider:
    """Runtime-shape shim (provider class name / last_outcome bookkeeping). It performs no decision-making."""

    def __init__(self, adapter: HumanParticipantPilotAdapter):
        self.adapter = adapter
        self.last_outcome: Optional[Dict[str, Any]] = None

    def generate_plan(self, request_packet):  # pragma: no cover
        raise RuntimeError("a human participant has no generate_plan path")


# ------------------------------------------------------------------------------------------------------------------------------------
# Research telemetry
# ------------------------------------------------------------------------------------------------------------------------------------
class HumanTelemetry:
    """Scientific action/event trace for ONE human agent (`human_participant_events.jsonl`) plus a separate low-level input trace
    (`human_input_trace.jsonl`). Events also go to the simulator event stream so the observer GUI sees them.

    Never recorded: raw keystrokes, unsent chat drafts. The low-level trace holds pointer clicks only and is not a measure of cognition."""

    EVENT_SCHEMA = "human_participant_event/1"
    INPUT_SCHEMA = "human_input_trace/1"

    def __init__(self, agent, controller: HumanParticipantController, clock=time.time):
        self.agent = agent
        self.controller = controller
        self._clock = clock
        self._open_sources: Dict[str, Dict[str, Any]] = {}

    # -- plumbing
    def _base(self, sim_state, event_type: str) -> Dict[str, Any]:
        wall = self._clock()
        return {
            "schema": self.EVENT_SCHEMA, "event_type": event_type, "participant_id": self.controller.participant_id, "agent_id": self.agent.agent_id,
            "agent": self.agent.name, "role": self.agent.role, "controller_type": CONTROLLER_TYPE,
            "sim_time": round(float(getattr(sim_state, "time", 0.0)), 3), "wall_ts": round(wall, 3),
            "wall_iso": datetime.fromtimestamp(wall, tz=timezone.utc).isoformat(timespec="milliseconds"),
        }

    def _write(self, sim_state, filename: str, row: Dict[str, Any]) -> None:
        try:
            path = Path(sim_state.logger.output_session.build_log_path(filename))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str, ensure_ascii=True) + "\n")
        except Exception as exc:  # observability must never break the run
            try:
                sim_state.logger.log_event(sim_state.time, "human_trace_write_failed", {"error": repr(exc), "file": filename})
            except Exception:
                pass

    def record(self, sim_state, event_type: str, *, intended: Optional[Dict[str, Any]] = None, grounded: Optional[Dict[str, Any]] = None,
               outcome: Optional[str] = None, blockers: Optional[List[str]] = None, public_message: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        row = self._base(sim_state, event_type)
        row.update({"intended_action": intended, "grounded_action": grounded, "outcome": outcome, "blockers": list(blockers or []), "public_message": public_message})
        row.update(extra)
        self._write(sim_state, "human_participant_events.jsonl", row)
        # Mirror to the observer event stream (the logger stamps its own time; payload carries the full research row).
        sim_state.logger.log_event(sim_state.time, f"human_{event_type}", {k: v for k, v in row.items() if k not in {"schema", "event_type"}})
        return row

    def record_input(self, sim_state, kind: str, **payload: Any) -> None:
        row = self._base(sim_state, "input")
        row.update({"schema": self.INPUT_SCHEMA, "kind": kind, **payload})
        self._write(sim_state, "human_input_trace.jsonl", row)

    # -- information source open/close (dwell). A source is "opened" when the simulator inspect interaction completes, "closed" by the participant.
    def source_opened(self, sim_state, source_id: str, *, element_ids: List[str], first_time: bool) -> None:
        if source_id in self._open_sources:
            return
        self._open_sources[source_id] = {"sim": float(sim_state.time), "wall": self._clock()}
        self.record(sim_state, "source_opened", outcome="opened", source_id=source_id, element_ids=list(element_ids), first_time=bool(first_time))

    def source_closed(self, sim_state, source_id: str, *, reason: str = "participant_closed") -> Optional[float]:
        opened = self._open_sources.pop(source_id, None)
        if opened is None:
            return None
        dwell_sim = float(sim_state.time) - opened["sim"]
        self.record(sim_state, "source_closed", outcome="closed", source_id=source_id, dwell_sim_s=round(dwell_sim, 3), dwell_wall_s=round(self._clock() - opened["wall"], 3), reason=reason)
        return dwell_sim

    def open_source_ids(self) -> List[str]:
        return sorted(self._open_sources)

    def flush(self, sim_state, reason: str = "session_end") -> None:
        for source_id in list(self._open_sources):
            self.source_closed(sim_state, source_id, reason=reason)
