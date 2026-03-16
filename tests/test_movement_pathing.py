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


if __name__ == "__main__":
    unittest.main()
