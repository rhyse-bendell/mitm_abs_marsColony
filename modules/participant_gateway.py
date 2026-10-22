"""Participant gateway: the narrow, thread-safe seam between a participant client (server thread) and the simulation (its own thread).

    client message --> ParticipantGateway.on_message --> HumanParticipantController queues (intents / UI events)  --> [sim thread] Mecha: ActionGate -> Environment
    sim thread     --> ParticipantGateway.publish    --> immutable JSON snapshot  --> latest()  --> client

The gateway never reads or writes simulation objects from the server thread: it only touches the controller queues and the last published snapshot.
`publish()` runs on the simulation thread (the host loop calls it after stepping) and is the only place a participant view is built.
A client can only (a) move, (b) pick an option the Middleware's affordances produced (by opaque id), (c) send typed chat, (d) report interface events
(menu opened, source panel closed/reopened). It cannot name an arbitrary action or supply a target: a forged item id is rejected as stale.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

from modules.action_schema import MAX_FREE_TEXT_CHARS, ExecutableActionType
from modules.human_participant import HumanIntent, HumanParticipantController
from modules.participant_view import build_participant_view, build_static_layout


class ParticipantGateway:
    def __init__(self, sim, agent_name: str, *, scheduler=None, publish_interval_s: float = 0.25, clock=time.time):
        self.sim = sim
        self.agent = next(a for a in sim.agents if a.name == agent_name or a.role == agent_name)
        if not self.agent.is_human_participant():
            raise ValueError(f"{agent_name!r} is not controlled by a human_participant")
        self.controller: HumanParticipantController = self.agent.human_controller(sim)
        self.scheduler = scheduler
        self.publish_interval_s = float(publish_interval_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._latest: Tuple[int, str] = (0, "")
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._last_publish = 0.0
        self._last_feedback_id = 0
        self.layout_json = json.dumps({"t": "layout", "layout": build_static_layout(sim), "you": {"name": self.agent.name, "role": self.agent.role}})

    # ---- simulation thread -------------------------------------------------------------------------------------------------
    def publish(self, force: bool = False) -> None:
        now = self._clock()
        if not force and now - self._last_publish < self.publish_interval_s:
            return
        self._last_publish = now
        telemetry = getattr(self.agent, "_human_telemetry", None)
        registry: Dict[str, Dict[str, Any]] = {}
        view = build_participant_view(self.sim, self.agent, self.controller, telemetry, seq=self._seq + 1, paused=bool(self.scheduler and self.scheduler.paused),
                                      menu_registry=registry, last_feedback_id=0)
        text = json.dumps({"t": "state", "view": view}, default=str)
        with self._lock:
            self._seq += 1
            self._registry = registry
            self._latest = (self._seq, text)

    # ---- server thread -----------------------------------------------------------------------------------------------------
    def latest(self) -> Tuple[int, str]:
        with self._lock:
            return self._latest

    def on_connect(self) -> bool:
        """-> False if a participant client is already attached (one human, one client)."""
        with self.controller._lock:
            if self.controller.connected:
                return False
        self.controller.set_connected(True)
        self.controller.submit_ui_event({"kind": "client_connected"})
        return True

    def on_disconnect(self) -> None:
        self.controller.set_connected(False)

    @staticmethod
    def _clean_attachments(raw) -> list:
        if not isinstance(raw, list):
            return []
        out = []
        for v in raw[:8]:
            if isinstance(v, str) and 0 < len(v) <= 80 and v not in out:
                out.append(v)
        return out

    def _reject(self, reason: str) -> Dict[str, Any]:
        return {"t": "ack", "ok": False, "reason": reason}

    def on_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        kind = msg.get("t")
        c = self.controller
        c.touch()
        if kind == "ping":
            return {"t": "pong"}
        if kind == "move":
            x, y = msg.get("x"), msg.get("y")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
                return self._reject("bad_coordinates")
            res = c.submit_intent(HumanIntent(action=ExecutableActionType.MOVE_TO_TARGET.value, action_args={"x": float(x), "y": float(y)}, source="map_click", label="move", clicked_at=[float(x), float(y)]))
            return {"t": "ack", "ok": res["accepted"], "reason": res["reason"]}
        if kind == "move_object":
            obj = str(msg.get("object") or "")
            if not obj or len(obj) > 80:
                return self._reject("bad_object")
            # Destination resolution reads world geometry, so it happens on the simulation thread (agent._process_human_intent).
            res = c.submit_intent(HumanIntent(action=ExecutableActionType.MOVE_TO_TARGET.value, action_args={}, source="map_click", ui_object=obj, label=f"move to {obj}"))
            return {"t": "ack", "ok": res["accepted"], "reason": res["reason"]}
        if kind == "action":
            item_id = str(msg.get("item") or "")
            if item_id.startswith("reopen:"):
                c.submit_ui_event({"kind": "reopen_source", "source": item_id.split(":", 1)[1]})
                return {"t": "ack", "ok": True, "reason": None}
            with self._lock:
                intent = self._registry.get(item_id)
            if intent is None:
                c.push_feedback("blocked", "That option is no longer available.", kind="action")
                return self._reject("stale_option")
            args = dict(intent["action_args"])
            if intent["action"] in {ExecutableActionType.START_CONSTRUCTION.value, ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value}:
                design = msg.get("design")
                if isinstance(design, dict) and design:
                    args["design"] = {str(k)[:40]: (v if isinstance(v, (str, int)) and not isinstance(v, bool) else str(v)[:60]) for k, v in list(design.items())[:16]}
            if intent["action"] == ExecutableActionType.EXTERNALIZE_PLAN.value:
                text = str(msg.get("text") or "").strip()
                if not text:
                    c.push_feedback("blocked", "Type a note first.", kind="whiteboard")
                    return self._reject("empty_message")  # an empty note is not an event
                args["text"] = text[:280]
                attachments = self._clean_attachments(msg.get("attachments"))
                if attachments:
                    args["attachments"] = attachments
            res = c.submit_intent(HumanIntent(action=intent["action"], target_id=intent["target_id"], action_args=args, source="menu", ui_object=intent.get("ui_object") or None, label=intent.get("label", "")))
            return {"t": "ack", "ok": res["accepted"], "reason": res["reason"]}
        if kind == "chat":
            text = str(msg.get("text") or "").strip()
            if not text:
                c.push_feedback("blocked", "Type a message first.", kind="chat")
                return self._reject("empty_message")  # nothing is logged: an unsent/empty draft is not an event
            if len(text) > MAX_FREE_TEXT_CHARS:
                c.push_feedback("blocked", "That message is too long.", kind="chat")
                return self._reject("message_too_long")
            chat_args = {"text": text, "scope": "team"}
            attachments = self._clean_attachments(msg.get("attachments"))
            if attachments:
                chat_args["attachments"] = attachments  # DIK element ids the participant ticked; the gate refuses any they have not encountered
            res = c.submit_intent(HumanIntent(action=ExecutableActionType.COMMUNICATE.value, action_args=chat_args, source="chat", label="team chat"))
            return {"t": "ack", "ok": res["accepted"], "reason": res["reason"]}
        if kind == "menu_opened":
            c.submit_ui_event({"kind": "menu_opened", "object": str(msg.get("object") or "")[:80]})
            return None
        if kind == "close_source":
            c.submit_ui_event({"kind": "close_source", "source": str(msg.get("source") or "")[:80]})
            return None
        if kind == "feedback_since":
            return {"t": "feedback", "items": c.feedback_since(int(msg.get("id") or 0))}
        return self._reject("unknown_message")
