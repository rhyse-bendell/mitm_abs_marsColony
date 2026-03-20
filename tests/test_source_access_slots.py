import unittest
from unittest.mock import patch

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState


class _Logger:
    def __init__(self):
        self.events = []

    def log_event(self, t, event_type, payload):
        self.events.append((event_type, payload))


class _Sim:
    def __init__(self):
        self.time = 0.0
        self.logger = _Logger()


class TestSourceAccessSlots(unittest.TestCase):
    def test_multiple_agents_target_different_team_info_slots(self):
        env = Environment(phases=[])
        s1 = env.select_source_access_point("Team_Info", agent_id="A1", from_position=(6.0, 1.0))
        s2 = env.select_source_access_point("Team_Info", agent_id="A2", from_position=(5.0, 1.0))
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertEqual(s1["kind"], "slot")
        self.assertEqual(s2["kind"], "slot")
        self.assertNotEqual(s1["slot_id"], s2["slot_id"])

    def test_occupied_slot_prefers_alternate_slot(self):
        env = Environment(phases=[])
        first = env.select_source_access_point("Team_Info", agent_id="A1", from_position=(8.5, 5.0))
        self.assertEqual(first["kind"], "slot")
        second = env.select_source_access_point("Team_Info", agent_id="A2", from_position=(8.5, 5.0))
        self.assertEqual(second["kind"], "slot")
        self.assertNotEqual(first["slot_id"], second["slot_id"])

    def test_all_slots_occupied_returns_queue_target(self):
        env = Environment(phases=[])
        slots = env.get_source_access_slots("Team_Info")
        for idx, slot in enumerate(slots):
            env.reserve_source_access_slot("Team_Info", slot["slot_id"], f"A{idx}")

        queued = env.select_source_access_point("Team_Info", agent_id="WAITER", from_position=(8.0, 5.0))
        self.assertIsNotNone(queued)
        self.assertEqual(queued["kind"], "queue")

    def test_near_source_is_not_usable_access_without_slot(self):
        env = Environment(phases=[])
        near_center = (8.0, 6.5)
        self.assertTrue(env.can_access_info(near_center, "Team_Info", role="Engineer"))
        usable, reason = env.can_agent_use_source_slot("Team_Info", "A1", near_center, role="Engineer")
        self.assertFalse(usable)
        self.assertEqual(reason, "not_at_interaction_slot")

    def test_unstuck_backoff_triggers_after_repeated_blocked_attempts(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=(8.0, 6.5), agent_id="A1")
        sim = _Sim()

        for i in range(5):
            sim.time = float(i)
            agent._inspect_source(env, "Team_Info", sim_state=sim)

        event_types = [e[0] for e in sim.logger.events]
        self.assertIn("source_access_unstuck_backoff", event_types)
        self.assertIn("source_slot_released", event_types)

    def test_smoke_simulation_still_runs(self):
        sim = SimulationState(phases=[])
        sim.update(0.2)
        sim.update(0.2)
        self.assertGreater(sim.time, 0.0)
        sim.stop()

    def test_shared_source_access_not_credited_before_valid_arrival(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=(8.0, 6.5), agent_id="A1")
        sim = _Sim()

        success = agent._inspect_source(env, "Team_Info", sim_state=sim)
        self.assertFalse(success)
        event_types = [e[0] for e in sim.logger.events]
        self.assertNotIn("inspect_started", event_types)
        self.assertNotIn("source_access_succeeded", event_types)
        self.assertIn("source_access_legality_checked", event_types)

    def test_role_private_source_requires_role_and_valid_slot(self):
        env = Environment(phases=[])
        wrong_role_agent = Agent(name="Engineer", role="Engineer", position=(3.45, 0.92), agent_id="A1")
        sim = _Sim()

        success = wrong_role_agent._inspect_source(env, "Architect_Info", sim_state=sim)
        self.assertFalse(success)
        events = sim.logger.events
        event_types = [e[0] for e in events]
        self.assertNotIn("source_access_succeeded", event_types)
        legality = [p for et, p in events if et == "source_access_legality_checked"]
        self.assertTrue(legality)
        self.assertFalse(legality[-1].get("access_legal"))

    def test_bootstrap_cannot_bypass_physical_legality(self):
        env = Environment(phases=[])
        agent = Agent(name="Architect", role="Architect", position=(8.0, 6.5), agent_id="A1")
        agent.activate_fallback_bootstrap(reason="test")
        sim = _Sim()

        for i in range(5):
            sim.time = float(i)
            success = agent._inspect_source(env, "Team_Info", sim_state=sim)
            self.assertFalse(success)
        event_types = [e[0] for e in sim.logger.events]
        self.assertNotIn("fallback_bootstrap_source_access_override", event_types)
        self.assertNotIn("source_access_succeeded", event_types)

    def test_successful_inspect_emits_reconstructable_handshake_sequence(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=(0.0, 0.0), agent_id="A1")
        sim = _Sim()
        selected = agent._select_source_access_target(env, "Team_Info", sim_state=sim)
        self.assertIsNotNone(selected)
        agent.position = selected["position"]

        with patch("modules.agent.random.random", return_value=0.0):
            success = agent._inspect_source(env, "Team_Info", sim_state=sim)
        self.assertTrue(success)
        event_types = [e[0] for e in sim.logger.events]
        self.assertIn("source_slot_selected", event_types)
        self.assertIn("source_access_arrival_confirmed", event_types)
        self.assertIn("source_access_legality_checked", event_types)
        self.assertIn("inspect_started", event_types)
        self.assertIn("source_access_succeeded", event_types)


if __name__ == "__main__":
    unittest.main()
