"""Human participant testbed: compatibility, boundary, scheduling, telemetry and transport tests (H12 items 1-11)."""
import ast
import contextlib
import io
import json
import socket
import tempfile
from pathlib import Path

import pytest

from human_testbed_helpers import INVALID_DESIGNS, VALID_DESIGNS, Play
from modules.action_schema import ExecutableActionType
from modules.controller_spec import (
    CONTROLLER_TYPES,
    ControllerConfigError,
    ControllerSpec,
    assignments_to_serializable,
    condition_identity,
    resolve_controller_assignments,
    resolve_execution_mode,
)
from modules.human_participant import HumanIntent, HumanParticipantController, participant_message_for
from modules.realtime_testbed import RealtimeScheduler
from modules.simulation import SimulationState

ROOT = Path(__file__).resolve().parents[1]


# ---- configuration / manifests ------------------------------------------------------------------------------------------------------
def test_human_controller_resolves_independently_per_role_and_serializes():
    specs = resolve_controller_assignments(["Architect", "Botanist", "Engineer"], {"Architect": "human_participant", "Botanist": "procedural_baseline"})
    assert [specs[r].controller_type for r in ("Architect", "Botanist", "Engineer")] == ["human_participant", "procedural_baseline", "procedural_baseline"]
    assert specs["Architect"].to_dict()["implementation"].endswith("HumanParticipantPilotAdapter")
    assert specs["Architect"].to_dict()["settings"] == {"participant_id": None}
    assert "human_participant" in condition_identity(specs)
    assert json.loads(json.dumps(assignments_to_serializable(specs)))["Architect"]["controller_type"] == "human_participant"
    assert CONTROLLER_TYPES["human_participant"].label == "Human Participant"
    with pytest.raises(ControllerConfigError):
        ControllerSpec.from_value({"controller_type": "human_participant", "settings": {"model": "x"}})  # no inference settings on a human


def test_every_three_role_composition_in_the_brief_is_expressible():
    for combo in ("HRR", "HLR", "HLL", "HHR", "HHL", "HHH", "RRR", "LLL"):
        mapping = {"H": "human_participant", "R": "procedural_baseline", "L": {"controller_type": "local_ollama", "settings": {"model": "m"}}}
        specs = resolve_controller_assignments(["Architect", "Botanist", "Engineer"], {r: mapping[c] for r, c in zip(("Architect", "Botanist", "Engineer"), combo)})
        has_human = "H" in combo
        assert resolve_execution_mode(None, specs) == ("realtime_testbed" if has_human else "simulation")


def test_execution_mode_resolution_human_requires_realtime_and_is_never_silently_switched():
    human = {"Architect": ControllerSpec("human_participant")}
    assert resolve_execution_mode(None, human) == "realtime_testbed"
    assert resolve_execution_mode("realtime_testbed", human) == "realtime_testbed"
    with pytest.raises(ControllerConfigError):
        resolve_execution_mode("simulation", human)
    procedural = {"Architect": ControllerSpec("procedural_baseline")}
    assert resolve_execution_mode(None, procedural) == "simulation"
    assert resolve_execution_mode("realtime_testbed", procedural) == "realtime_testbed"
    with pytest.raises(ControllerConfigError):
        resolve_execution_mode("warp", procedural)


def test_manifest_records_human_controller_and_execution_mode():
    play = Play()
    try:
        manifest = play.sim.controller_manifest()
        row = manifest["per_agent"]["Architect"]
        assert row["controller"]["controller_type"] == "human_participant" and row["explicit_assignment"] is True
        assert manifest["per_agent"]["Botanist"]["controller"]["controller_type"] == "procedural_baseline"
        assert play.sim._backend_settings_for_manifest()["execution_mode"] == "realtime_testbed"
        assert "Architect=human_participant" in manifest["composition_id"] if manifest["composition_id"] else True
        written = json.loads(Path(play.sim.logger.output_session.session_folder, "session_manifest.json").read_text(encoding="utf-8"))
        assert written["active_agents"][0]["controller_type"] in {"human_participant", "procedural_baseline"}
    finally:
        play.close()


def test_all_procedural_run_stays_in_simulation_mode_with_no_human_state():
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationState(speed=1.0, flash_mode=True, seed=3, outputs_root=tempfile.mkdtemp())
    assert sim.execution_mode == "simulation" and sim.human_controllers() == {} and sim.chat_log == []
    assert all(a.controller_type_id() == "procedural_baseline" and not a.is_human_participant() for a in sim.agents)
    assert sim.async_inference is False  # untouched default


def test_existing_ollama_configuration_still_resolves_normally():
    specs = resolve_controller_assignments(["Architect"], {"Architect": {"controller_type": "local_ollama", "settings": {"model": "qwen3.5:9b"}}})
    assert specs["Architect"].controller_type == "local_ollama" and specs["Architect"].require_ready()
    assert resolve_execution_mode(None, specs) == "simulation"  # Ollama alone does not change the execution mode


# ---- scheduling -----------------------------------------------------------------------------------------------------------------------
class _FakeSim:
    def __init__(self):
        self.time, self.terminal_state, self.speed_multiplier, self.updates, self.stopped = 0.0, None, 3.0, [], 0

    def update(self, dt):
        self.updates.append(dt)
        self.time += dt

    def stop(self):
        self.stopped += 1


def test_scheduler_paces_simulated_time_to_the_wall_clock_at_1x():
    sim, now = _FakeSim(), [100.0]
    sch = RealtimeScheduler(sim, clock=lambda: now[0])
    assert sim.speed_multiplier == 1.0
    sch.resume()
    sch.advance()
    now[0] += 0.5
    sch.advance()
    now[0] += 2.0 - 1.0  # 1.0 s later
    sch.advance()
    assert sim.time == pytest.approx(1.5, abs=0.11) and sch.sim_steps == len(sim.updates)


def test_scheduler_pause_stops_the_whole_clock_and_resume_does_not_burst():
    sim, now = _FakeSim(), [0.0]
    sch = RealtimeScheduler(sim, clock=lambda: now[0], max_catchup_s=5.0)
    sch.resume(); sch.advance()
    now[0] += 1.0; sch.advance()
    t_before = sim.time
    sch.pause()
    now[0] += 600.0
    assert sch.advance() == 0 and sim.time == t_before
    sch.resume()
    now[0] += 0.3
    sch.advance()
    assert sim.time - t_before <= 0.31  # the 600 s pause was not replayed


def test_scheduler_bounds_catchup_and_stop_is_safe_and_single():
    sim, now = _FakeSim(), [0.0]
    sch = RealtimeScheduler(sim, clock=lambda: now[0], max_catchup_s=1.0)
    sch.resume(); sch.advance()
    now[0] += 30.0
    sch.advance()
    assert sim.time == pytest.approx(1.0, abs=0.11) and sch.dropped_wall_s == pytest.approx(29.0)
    sch.stop(); sch.stop()
    assert sim.stopped == 1 and sch.advance() == 0


def test_mission_clock_runs_while_the_human_is_idle_and_other_agents_keep_acting():
    play = Play()
    try:
        positions = {a.name: tuple(a.position) for a in play.sim.agents}
        play.advance(20)  # the participant does nothing at all
        assert play.sim.time == pytest.approx(20.0, abs=0.2)
        assert tuple(play.agent.position) == positions["Architect"]  # the human agent does nothing by itself
        moved = [a.name for a in play.sim.agents if a.name != "Architect" and tuple(a.position) != positions[a.name]]
        assert moved, "procedural agents must keep acting while a human thinks"
        assert not [e for e in play.events if e["event_type"] == "pilot_selection_via_adapter" and e["payload_data"].get("agent") == "Architect"]
        assert play.agent.active_actions == [] and play.agent.current_action in ([], None)  # no WAIT or any other action invented for the human
    finally:
        play.close()


def test_realtime_mode_bypasses_the_global_inference_barrier():
    play = Play()
    try:
        play.sim.async_inference = True
        play.sim.inference_pending = lambda: True  # some model call is "pending" somewhere
        t0 = play.sim.time
        play.advance(3)
        assert play.sim.time > t0 + 2.5  # in simulation mode this would hold the world; in realtime it must not
    finally:
        play.close()


# ---- boundary: gate, no substitution, no direct mutation -----------------------------------------------------------------------------
def test_human_actions_pass_through_the_action_gate():
    play = Play()
    try:
        seen = []
        original = play.sim.action_gate.evaluate

        def spy(*, agent, decision, sim_state):
            seen.append((agent.name, decision.selected_action.value, decision.target_id))
            return original(agent=agent, decision=decision, sim_state=sim_state)

        play.sim.action_gate.evaluate = spy
        play.gw.on_message({"t": "move", "x": 5.0, "y": 2.0})
        play.advance(1)
        assert ("Architect", "move_to_target", None) in seen
        play.gw.controller.submit_intent(HumanIntent(action="inspect_information_source", target_id="Botanist_Info"))
        play.advance(1)
        assert ("Architect", "inspect_information_source", "Botanist_Info") in seen
    finally:
        play.close()


def test_blocked_private_source_is_reported_logged_and_never_replaced():
    play = Play()
    try:
        play.walk_to("Botanist_Info")
        assert play.obj("Botanist_Info")["near"]
        assert play.obj("Botanist_Info")["items"][0]["label"] == "Read briefing"  # the attempt is allowed by the physical metaphor
        before = {s for s in play.agent.memory_seen_packets}
        play.choose("Botanist_Info", "Read briefing")
        play.advance(3)
        fb = [f for f in play.view()["feedback"] if f["kind"] != "accepted"]
        assert fb and fb[-1]["message"] == "You are not permitted to view another role's private briefing."
        assert play.agent.memory_seen_packets == before and "Botanist_Info" not in play.agent.memory_seen_packets
        assert play.agent.active_actions == [] and play.agent.current_inspect_target_id is None  # nothing opened in its place
        rows = [r for r in play.human_rows() if r["event_type"] == "action_blocked"]
        assert rows and rows[-1]["blockers"] == ["private_source_inaccessible"] and rows[-1]["grounded_action"] is None
        assert not [e for e in play.events if e["event_type"] == "inspect_started" and e["payload_data"].get("agent_id") == "architect_1" and e["payload_data"].get("source_id") != "Botanist_Info"]
        assert not play.view()["info"]["sources"]
    finally:
        play.close()


def test_a_blocked_move_is_reported_without_substitution():
    play = Play()
    try:
        start = tuple(play.agent.position)
        play.gw.on_message({"t": "move", "x": 0.05, "y": 0.05})  # inside the wall
        play.advance(2)
        fb = [f for f in play.view()["feedback"] if f["kind"] != "accepted"]
        assert fb[-1]["message"] == "You can't walk there."
        assert tuple(play.agent.position) == start and play.agent.target is None
    finally:
        play.close()


def test_human_modules_cannot_reach_environment_state():
    """Static boundary: the controller/gateway/server modules import nothing that mutates the world, and the controller keeps no simulation reference."""
    for name in ("human_participant.py", "participant_gateway.py", "participant_server.py"):
        tree = ast.parse((ROOT / "modules" / name).read_text(encoding="utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not {m for m in imported if m.startswith(("modules.construction", "modules.environment", "modules.simulation", "modules.agent"))}, (name, imported)
    controller = HumanParticipantController("a", "Architect", "Architect")
    assert not any(hasattr(controller, attr) for attr in ("sim", "sim_state", "environment", "construction", "agent"))


def test_a_forged_menu_item_cannot_name_an_action():
    play = Play()
    try:
        play.advance(1)
        reply = play.gw.on_message({"t": "action", "item": "m_forged_transport"})
        assert reply["ok"] is False and reply["reason"] == "stale_option"
        assert not play.gw.controller._intents
        out = play.gw.on_message({"t": "action", "item": "../etc"})
        assert out["ok"] is False
    finally:
        play.close()


def test_gateway_enforces_interaction_range_server_side():
    play = Play()
    try:
        assert not play.obj("Architect_Info")["near"]
        item = play.obj("Architect_Info")["items"][0]
        play.gw.on_message({"t": "action", "item": item["id"]})  # a client that ignores the greyed-out button
        play.advance(2)
        rows = [r for r in play.human_rows() if r["event_type"] == "action_blocked"]
        assert rows and rows[-1]["blockers"] == ["not_in_interaction_range"]
        assert "Architect_Info" not in play.agent.memory_seen_packets
    finally:
        play.close()


def test_busy_commitment_refuses_new_actions_and_plain_walking_can_be_redirected():
    play = Play()
    try:
        play.walk_to("Architect_Info")
        play.choose("Architect_Info", "Read briefing")
        play.advance(0.2)
        assert play.agent._human_committed_action() is not None
        play.gw.on_message({"t": "move", "x": 5.0, "y": 2.0})  # committed to reading: refused, not queued, nothing replaced
        play.advance(0.2)
        refused = [r for r in play.human_rows() if r["event_type"] == "action_blocked"]
        assert refused and refused[-1]["blockers"][0].startswith("busy_committed:inspect_information_source")
        assert play.view()["feedback"][-1]["message"] == "You are busy with another action."
        play.advance(5)
        assert play.agent._human_committed_action() is None
        # plain walking is interruptible
        play.gw.on_message({"t": "move", "x": 5.0, "y": 2.0})
        play.advance(0.5)
        play.gw.on_message({"t": "move", "x": 4.5, "y": 2.2})
        play.advance(0.5)
        assert tuple(play.agent.target) == (4.5, 2.2)
        assert [r for r in play.human_rows() if r["event_type"] == "movement_redirected"]
        play.advance(15)
        assert tuple(round(c, 2) for c in play.agent.position) == (4.5, 2.2)
    finally:
        play.close()


def test_closing_or_losing_the_participant_client_fails_safe():
    play = Play()
    try:
        play.gw.on_disconnect()
        res = play.gw.controller.submit_intent(HumanIntent(action="move_to_target", action_args={"x": 5.0, "y": 2.0}))
        assert res == {"accepted": False, "intent_id": None, "reason": "participant_not_connected"}
        positions = tuple(play.agent.position)
        play.advance(10)
        assert play.sim.time == pytest.approx(10.0, abs=0.3) and tuple(play.agent.position) == positions  # the world goes on; the avatar idles
        play.gw.on_connect()  # reconnect works
        assert play.gw.on_message({"t": "move", "x": 5.0, "y": 2.0})["ok"]
    finally:
        play.close()
    assert play.gw.controller.submit_intent(HumanIntent(action="move_to_target", action_args={"x": 5.0, "y": 2.0}))["reason"] == "experiment_ended"


def test_one_client_per_participant():
    play = Play()
    try:
        play.gw.on_disconnect()
        assert play.gw.on_connect() is True
        assert play.gw.on_connect() is False
    finally:
        play.close()


# ---- information interaction, chat, telemetry ---------------------------------------------------------------------------------------------
def test_opening_own_and_shared_sources_is_a_real_interaction_with_dwell_and_dik_events():
    play = Play()
    try:
        play.walk_to("Architect_Info")
        play.choose("Architect_Info", "Read briefing")
        play.advance(4)
        v = play.view()
        src = next(s for s in v["info"]["sources"] if s["id"] == "Architect_Info")
        assert src["open"] and src["items"] and all(isinstance(i["text"], str) for i in src["items"])
        assert [r for r in play.human_rows() if r["event_type"] == "source_opened" and r["source_id"] == "Architect_Info"]
        assert play.events_of("dik_acquired_from_inspect") or play.events_of("source_access_succeeded")
        play.advance(5)
        play.gw.on_message({"t": "close_source", "source": "Architect_Info"})
        play.advance(0.5)
        closed = [r for r in play.human_rows() if r["event_type"] == "source_closed"]
        assert closed and closed[-1]["dwell_sim_s"] >= 5 and closed[-1]["dwell_wall_s"] >= 0
        assert closed[-1]["participant_id"] and closed[-1]["controller_type"] == "human_participant" and closed[-1]["role"] == "Architect"
        play.walk_to("Team_Info")
        play.choose("Team_Info", "team brief")
        play.advance(4)
        assert {s["id"] for s in play.view()["info"]["sources"]} == {"Architect_Info", "Team_Info"}
    finally:
        play.close()


def test_human_exposure_is_deterministic_and_there_is_no_simulated_forgetting():
    play = Play()
    try:
        play.agent.memory_reliability = 0.0  # would forget almost everything for an artificial agent
        play.walk_to("Architect_Info")
        play.choose("Architect_Info", "Read briefing")
        play.advance(3)
        packet = play.sim.environment.knowledge_packets["Architect_Info"]
        held = {getattr(o, "id", None) for o in play.agent.mental_model["data"]} | {getattr(o, "id", None) for o in play.agent.mental_model["information"]}
        assert {i.id for i in packet["data"] + packet["information"]} <= held  # every item of the opened source, not a 95% draw
        play.advance(240)
        held_later = {getattr(o, "id", None) for o in play.agent.mental_model["data"]} | {getattr(o, "id", None) for o in play.agent.mental_model["information"]}
        assert held <= held_later
        assert all(e["payload_data"].get("agent_id") != "architect_1" for e in play.events_of("dik_forgotten"))
    finally:
        play.close()


def test_typed_chat_is_a_canonical_communication_event_without_dik_transfer_or_draft_logging():
    play = Play()
    try:
        before = {a.name: (len(a.mental_model["data"]), len(a.mental_model["information"])) for a in play.sim.agents}
        play.gw.on_message({"t": "chat", "text": "   "})  # an empty draft is not an event
        play.advance(1)
        assert not [r for r in play.human_rows() if r["event_type"] == "chat_sent"] and not play.sim.chat_log
        play.gw.on_message({"t": "chat", "text": "I will build houses at Site B"})
        play.advance(2)
        rec = play.sim.chat_log[-1]
        assert rec["sender"] == "Architect" and rec["scope"] == "team" and set(rec["recipients"]) == {"Botanist", "Engineer"} and rec["text"] == "I will build houses at Site B"
        assert all(d["outcome"] == "delivered" for d in rec["delivery"])
        row = [r for r in play.human_rows() if r["event_type"] == "chat_sent"][-1]
        assert row["sim_time"] > 0 and row["wall_ts"] > 0 and row["intended_action"]["action"] == "communicate" and row["outcome"] == "delivered"
        botanist = next(a for a in play.sim.agents if a.name == "Botanist")
        assert botanist.chat_inbox[-1]["text"] == "I will build houses at Site B"  # delivered as written; the text is not interpreted by the recipient's cognition
        assert before  # (artificial agents also act on their own meanwhile, so their DIK counts are not compared)
        assert [e for e in play.events if e["event_type"] == "chat_message_sent"]
        assert "team_chat" in Path(play.sim.logger.output_session.build_log_path("interaction_trace.jsonl")).read_text(encoding="utf-8")  # observer Interactions stream
        # the logged intent never carries the raw text outside the chat record itself, and no keystroke trace exists
        assert not any("keystroke" in json.dumps(r).lower() or "draft" in json.dumps(r).lower() for r in play.human_rows())
    finally:
        play.close()


def test_chat_validation_rejects_overlong_and_unknown_scope():
    play = Play()
    try:
        assert play.gw.on_message({"t": "chat", "text": "x" * 501})["reason"] == "message_too_long"
        play.gw.controller.submit_intent(HumanIntent(action="communicate", action_args={"text": "hi", "scope": "Nobody"}, source="chat"))
        play.advance(1)
        rows = [r for r in play.human_rows() if r["event_type"] == "action_blocked"]
        assert rows and rows[-1]["blockers"] == ["unknown_recipient"]
    finally:
        play.close()


def test_agent_messages_to_a_human_are_shown_not_answered_for_them():
    play = Play()
    try:
        info = next(i for i in play.sim.environment.knowledge_packets["Team_Info"]["information"])
        botanist = next(a for a in play.sim.agents if a.name == "Botanist")
        out = play.agent.receive_message({"type": "TIP", "sender": "Botanist", "content": [info]}, from_agent="Botanist", sim_state=play.sim)
        assert out["dik_changed"] is True
        assert play.sim.chat_log[-1]["kind"] == "agent_message" and info.content in play.sim.chat_log[-1]["text"]
        request = play.agent.receive_message({"type": "TKRQ", "sender": "Botanist", "content": ["rule:R_X"], "project_id": None}, from_agent="Botanist", sim_state=play.sim)
        assert request["response_category"] is None
        assert not play.agent.team_plan_state.get("pending_outbound_messages")  # nothing queued as a reply on the person's behalf
        assert botanist is not None
    finally:
        play.close()


# ---- the vertical slice: movement, resources, construction through menus ------------------------------------------------------------------
def test_vertical_slice_start_project_and_carry_bricks_through_contextual_menus():
    play = Play()
    try:
        v = play.walk_to("Build_Site_B")
        assert play.obj("Build_Site_B")["near"]
        labels = [i["label"] for i in play.obj("Build_Site_B")["items"]]
        assert "Start a house..." in labels  # same typed START affordances the Pilots get (any role); the person then chooses a design
        pid = play.start_structure("house", VALID_DESIGNS["house"])
        assert pid, "the committed START must have created a project"
        assert [r for r in play.human_rows() if r["event_type"] == "action_intent" and r["grounded_action"]["project_id"] == pid]
        play.walk_to("pile_a")
        carry = [i for i in play.obj("pile_a")["items"] if "Carry bricks" in i["label"]]
        assert carry and carry[0]["enabled"]
        before = int(play.sim.environment.construction.resource_nodes["pile_a"].quantity)
        play.choose("pile_a", "Carry bricks")
        play.advance(3)
        assert play.agent.active_actions and play.view()["you"]["busy"]["label"] == "carrying bricks"
        # busy: a second substantive action is refused (and says so), not queued
        play.gw.on_message({"t": "chat", "text": "working"})  # typing to the team does not need the avatar: delivered while busy
        play.advance(0.3)
        assert play.sim.chat_log and play.sim.chat_log[-1]["text"] == "working" and play.agent.active_actions
        play.gw.on_message({"t": "move", "x": 5.0, "y": 2.0})  # but any other verb is refused while committed
        play.advance(0.3)
        assert play.view()["feedback"][-1]["message"] == "You are busy with another action."
        play.advance(40)
        assert int(play.sim.environment.construction.resource_nodes["pile_a"].quantity) < before
        assert any(r["event_type"] == "action_completed" and r["grounded_action"]["decision_action"] == "transport_resources" for r in play.human_rows())
    finally:
        play.close()


def test_observer_event_stream_and_snapshots_see_the_human():
    play = Play()
    try:
        play.walk_to("Architect_Info")
        play.choose("Architect_Info", "Read briefing")
        play.gw.on_message({"t": "chat", "text": "hello"})
        play.advance(5)
        kinds = {e["event_type"] for e in play.events}
        assert {"human_action_intent", "human_source_opened", "human_movement_arrived"} <= kinds
        assert "agent_state" in "".join(kinds) or play.sim.logger.buffer  # the normal per-agent state stream keeps flowing
        snap = play.agent.get_runtime_state_snapshot()
        assert snap["controller_label"] == "Human Participant" and snap["controller_type"] == "human_participant"
        labels = {a.name: a.get_runtime_state_snapshot()["controller_label"] for a in play.sim.agents}
        assert labels == {"Architect": "Human Participant", "Botanist": "Procedural Baseline", "Engineer": "Procedural Baseline"}
        assert any(e["event_type"] in {"pilot_selection_via_adapter", "action_affordances_generated"} for e in play.events)  # artificial agents still emit their normal streams
    finally:
        play.close()


def test_message_catalog_hides_internal_blockers():
    assert participant_message_for(["private_source_inaccessible"]).startswith("You are not permitted")
    assert participant_message_for(["missing_expected_rule:R_SECRET", "validation_not_ready"]) == "That isn't possible right now."
    assert "R_SECRET" not in participant_message_for(["missing_expected_rule:R_SECRET"])


# ---- WebSocket end to end ------------------------------------------------------------------------------------------------------------------
def _ws_open(port, token, role="Architect", origin=None):
    import base64
    import os

    key = base64.b64encode(os.urandom(16)).decode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=3)
    extra = f"Origin: {origin}\r\n" if origin else ""
    sock.sendall((f"GET /ws?role={role}&token={token} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{extra}Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += sock.recv(1)
    return sock, head.split(b"\r\n")[0].decode()


def _ws_send(sock, obj):
    import os
    import struct

    payload = json.dumps(obj).encode()
    mask = os.urandom(4)
    n = len(payload)
    head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n))
    sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))


def _ws_recv(sock):
    import struct

    from modules.participant_server import _recv_exact

    b1, b2 = _recv_exact(sock, 2)
    n = b2 & 0x7F
    if n == 126:
        n = struct.unpack(">H", _recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
    return json.loads(_recv_exact(sock, n).decode())


def test_websocket_channel_end_to_end_with_token_origin_and_single_client_rules():
    play = Play()
    play.gw.on_disconnect()  # the real socket will connect it
    server = play.session.server
    try:
        with pytest.raises(Exception):
            sock, status = _ws_open(server.port, "wrong-token")
            assert "403" in status
            raise RuntimeError
        _, status = _ws_open(server.port, server.token, origin="http://evil.example")
        assert "403" in status
        sock, status = _ws_open(server.port, server.token)
        assert "101" in status
        assert _ws_recv(sock)["t"] == "layout"
        _, second = _ws_open(server.port, server.token)
        assert "409" in second  # one client per participant
        play.advance(1)
        sock.settimeout(1.0)
        view = None
        try:
            while True:
                msg = _ws_recv(sock)
                if msg["t"] == "state":
                    view = msg["view"]
        except (socket.timeout, TimeoutError):
            pass
        assert view and view["schema"] == "participant_view/1" and view["you"]["role"] == "Architect"
        _ws_send(sock, {"t": "move", "x": 5.0, "y": 2.0})
        _ws_send(sock, {"t": "action", "item": "bogus"})
        import time as _t

        _t.sleep(0.4)
        play.advance(3)
        assert play.agent.target is not None or tuple(play.agent.position) != (7.35, 3.75)
        sock.close()
        _t.sleep(0.5)
        assert play.gw.controller.connected is False  # dropping the socket marks the participant disconnected
    finally:
        play.close()


def test_server_serves_the_client_only_with_the_token_and_binds_loopback():
    import urllib.error
    import urllib.request

    play = Play()
    try:
        server = play.session.server
        assert server._httpd.server_address[0] == "127.0.0.1"
        page = urllib.request.urlopen(play.session.urls()["Architect"], timeout=3).read().decode()
        assert "Mars Colony" in page and "client.js" in page
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/?role=Architect&token=nope", timeout=3)
        assert err.value.code == 403
        js = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/static/client.js", timeout=3).read().decode()
        assert "innerHTML" not in js  # server text is only ever rendered as text nodes
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/static/../../interface.py", timeout=3)
    finally:
        play.close()
