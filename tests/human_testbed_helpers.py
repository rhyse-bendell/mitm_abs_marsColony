"""Shared helpers for the human participant testbed tests (not a test module)."""
import contextlib
import io
import json
import tempfile

from modules.realtime_testbed import HumanTestbedSession
from modules.simulation import SimulationState


class Play:
    """A realtime human session driven by a fake wall clock, so tests control exactly how much wall time passes."""

    def __init__(self, assignments=None, seed=3, role="Architect"):
        self.outputs = tempfile.mkdtemp(prefix="human_testbed_")
        assignments = assignments if assignments is not None else {role: "human_participant"}
        with contextlib.redirect_stdout(io.StringIO()):
            self.sim = SimulationState(speed=1.0, seed=seed, outputs_root=self.outputs, controller_assignments=assignments)
        self.now = [0.0]
        self.session = HumanTestbedSession(self.sim, clock=lambda: self.now[0]).start()
        self.role = role
        self.gw = self.session.gateways.get(role)
        self.agent = next(a for a in self.sim.agents if a.role == role)
        if self.gw:
            self.gw.controller.set_connected(True)
        self.events = []
        self.sim.logger.register_event_listener(self.events.append)

    def advance(self, wall_s, publish=True):
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(int(round(wall_s * 10))):
                self.now[0] += 0.1
                self.session.tick()
        if publish and self.gw:
            self.gw.publish(force=True)

    def view(self):
        return json.loads(self.gw.latest()[1])["view"]

    def obj(self, object_id):
        return next(o for o in self.view()["objects"] if o["id"] == object_id)

    def walk_to(self, object_id, limit_s=90):
        self.gw.on_message({"t": "move_object", "object": object_id})
        for _ in range(int(limit_s / 2)):
            self.advance(2)
            v = self.view()
            if self.obj(object_id)["near"] and not v["you"]["moving"]:
                return v
        return self.view()

    def choose(self, object_id, label_part):
        item = next(i for i in self.obj(object_id)["items"] if label_part in i["label"])
        return self.gw.on_message({"t": "action", "item": item["id"]})

    def events_of(self, event_type):
        return [e for e in self.events if e["event_type"] == event_type]

    def human_rows(self):
        path = self.sim.logger.output_session.build_log_path("human_participant_events.jsonl")
        try:
            return [json.loads(line) for line in open(path, encoding="utf-8")]
        except FileNotFoundError:
            return []

    def close(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.session.stop()
