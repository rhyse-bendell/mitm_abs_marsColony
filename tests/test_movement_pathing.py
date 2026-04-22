import unittest

from modules.agent import Agent
from modules.environment import Environment
from modules.simulation import SimulationState


class TestMovementPathing(unittest.TestCase):
    def test_path_can_be_planned_and_reused(self):
        env = Environment(phases=[])
        start = env.get_spawn_point("Engineer")
        target = env.get_interaction_target_position("Team_Info", from_position=start)
        self.assertIsNotNone(target)

        first = env.plan_path(start, target, mode="grid_astar")
        second = env.plan_path(start, target, mode="grid_astar")

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertTrue(second["from_cache"])
        self.assertGreaterEqual(len(second["waypoints"]), 1)

    def test_unreachable_target_classification(self):
        env = Environment(phases=[])
        blocked = env.objects["Blocked_Zone_AC"]["corners"]
        (x1, y1), (x2, y2) = blocked
        inside_block = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        result = env.plan_path(env.get_spawn_point("Engineer"), inside_block, mode="grid_astar")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["blocker_category"], "target_unreachable")

    def test_zero_distance_retarget_loops_are_classified(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.get_spawn_point("Engineer"))
        target = tuple(agent.position)
        agent.navigation["last_target"] = (agent.position[0] + 0.5, agent.position[1])
        events = []

        class _Logger:
            def log_event(self, t, event_type, payload):
                events.append((event_type, payload))

        class _Sim:
            time = 0.0
            logger = _Logger()

        agent.move_toward(target, dt=1.0, environment=env, sim_state=_Sim())
        self.assertTrue(any(e[0] == "movement_failed" and e[1].get("failure_category") == "zero_distance_retarget" for e in events))

    def test_collision_disable_option_changes_blocking(self):
        env = Environment(phases=[])
        a1 = Agent(name="A1", role="Engineer", position=(4.0, 1.0), planner_config={"ignore_agent_collision": False})
        a2 = Agent(name="A2", role="Engineer", position=(4.2, 1.0), planner_config={"ignore_agent_collision": False})
        env.agents = [a1, a2]
        before = tuple(a1.position)
        a1.move_toward((4.2, 1.0), dt=1.0, environment=env)
        self.assertEqual(before, a1.position)

        b1 = Agent(name="B1", role="Engineer", position=(4.0, 1.0), planner_config={"ignore_agent_collision": True})
        b2 = Agent(name="B2", role="Engineer", position=(4.2, 1.0), planner_config={"ignore_agent_collision": True})
        env.agents = [b1, b2]
        before_b = tuple(b1.position)
        b1.move_toward((4.2, 1.0), dt=1.0, environment=env)
        self.assertNotEqual(before_b, b1.position)

    def test_movement_started_to_arrived_progression(self):
        env = Environment(phases=[])
        start = env.get_spawn_point("Engineer")
        target = env.get_interaction_target_position("Engineer_Info", from_position=start)
        self.assertIsNotNone(target)
        agent = Agent(name="Engineer", role="Engineer", position=start)

        events = []

        class _Logger:
            def log_event(self, t, event_type, payload):
                events.append(event_type)

        class _Sim:
            time = 0.0
            logger = _Logger()

        for _ in range(50):
            _Sim.time += 0.2
            agent.move_toward(target, dt=0.2, environment=env, sim_state=_Sim())

        self.assertIn("movement_started", events)
        self.assertIn("movement_progressed", events)
        self.assertIn("movement_arrived", events)

    def test_runtime_witness_audit_specific_movement_category(self):
        sim = SimulationState(phases=[])
        sim.logger.log_event(sim.time, "movement_blocked", {"agent": sim.agents[0].name, "blocker_category": "no_path_found"})
        result = sim.runtime_witness_audit.finalize()
        categories = result["summary"]["witness_step_failures_by_category"]
        self.assertIn("no_path_found", categories)
        sim.stop()

    def test_repeated_blocked_zone_blacklists_slot_and_clears_target(self):
        class _Logger:
            def __init__(self):
                self.events = []

            def log_event(self, t, event_type, payload):
                self.events.append((event_type, payload))

        class _Sim:
            def __init__(self):
                self.time = 0.0
                self.logger = _Logger()

        class _BlockedEnv:
            def __init__(self):
                self.objects = {"Blocked_Test": {"type": "blocked", "passable": False}}
                self.agents = []
                self.invalidate_calls = 0

            def plan_path(self, start, target, mode="grid_astar"):
                return {"status": "ok", "waypoints": [target], "from_cache": True, "path_mode": mode, "blocker_category": None}

            def is_near_object(self, point, name, threshold=0.15):
                return True

            def release_source_access_slot(self, packet_name, agent_id=None, slot_id=None):
                return [slot_id] if slot_id is not None else []

            def invalidate_path_cache_entry(self, start, target, mode="grid_astar", grid_step=0.35):
                self.invalidate_calls += 1
                return True

        env = _BlockedEnv()
        agent = Agent(name="Architect", role="Architect", position=(1.0, 1.0), agent_id="A1")
        sim = _Sim()
        env.agents = [agent]
        agent.current_inspect_target_id = "Team_Info"
        agent.source_access_state.update({"source_id": "Team_Info", "slot_id": "top_left", "slot_position": (2.0, 1.0), "target_kind": "slot"})
        agent._commit_inspect_pursuit("Team_Info", (2.0, 1.0), now_ts=0.0, slot_id="top_left", sim_state=sim)

        for i in range(3):
            sim.time = float(i)
            agent.move_toward((2.0, 1.0), dt=0.5, environment=env, sim_state=sim)

        events = [e[0] for e in sim.logger.events]
        self.assertIn("movement_target_temporarily_blacklisted", events)
        self.assertIn("movement_retarget_after_repeated_blocked_zone", events)
        self.assertIn("movement_cached_path_invalidated_after_blocked_zone", events)
        self.assertEqual(env.invalidate_calls, 1)
        self.assertIsNone(agent.target)
        self.assertIsNone(agent.inspect_pursuit.get("source_id"))


if __name__ == "__main__":
    unittest.main()
