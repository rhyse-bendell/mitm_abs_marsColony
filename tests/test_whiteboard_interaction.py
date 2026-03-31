import unittest

from modules.action_schema import ExecutableActionType
from modules.environment import Environment
from modules.simulation import SimulationState
from modules.task_model import load_task_model


class TestWhiteboardInteractionContract(unittest.TestCase):
    def test_task_model_includes_whiteboard_runtime_contract(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)

        self.assertIn("Whiteboard", env.objects)
        self.assertIn("Zone_Whiteboard", env.zones)
        self.assertIn("whiteboard", env.interaction_targets)
        self.assertEqual(env.interaction_targets["whiteboard"]["kind"], "artifact")

    def test_whiteboard_accessible_in_zone_and_near_surface(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)

        in_zone = env.get_interaction_access((6.2, 6.6), "whiteboard", role="Architect")
        near_zone = env.get_interaction_access((6.2, 6.2), "whiteboard", role="Architect")

        self.assertTrue(in_zone["accessible"])
        self.assertEqual(in_zone["reason"], "in_artifact_zone")
        self.assertTrue(near_zone["accessible"])
        self.assertEqual(near_zone["reason"], "near_artifact_surface")

    def test_whiteboard_target_position_is_reachable(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)

        start = env.get_spawn_point("Architect")
        target = env.get_interaction_target_position("whiteboard", from_position=start)

        self.assertIsNotNone(target)
        self.assertTrue(env.is_point_navigable(target))
        path = env.plan_path(start, target)
        self.assertTrue(path)

    def test_externalize_plan_execution_uses_whiteboard_without_unknown_target(self):
        sim = SimulationState(phases=[])
        try:
            agent = sim.agents[0]
            agent.position = (6.2, 6.6)
            action = {
                "type": "idle",
                "duration": 1.0,
                "priority": 1,
                "progress": 0.0,
                "artifact_action": ExecutableActionType.EXTERNALIZE_PLAN.value,
                "decision_action": ExecutableActionType.EXTERNALIZE_PLAN.value,
                "execution_stage": None,
            }
            agent.active_actions = [action]

            agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
            self.assertEqual(action.get("execution_stage"), "mutation_execution_succeeded")

            recent = sim.logger.get_recent_events(50)
            unknown = [
                e for e in recent
                if e.get("event_type") == "externalization_en_route"
                and e.get("payload_data", {}).get("access_reason") == "unknown_target"
            ]
            self.assertFalse(unknown)
        finally:
            sim.stop()

    def test_consult_team_artifact_can_resolve_and_access_whiteboard(self):
        sim = SimulationState(phases=[])
        try:
            agent = sim.agents[0]
            target = sim.environment.get_interaction_target_position("whiteboard", from_position=agent.position)
            self.assertIsNotNone(target)
            access = sim.environment.get_interaction_access(target, "whiteboard", role=agent.role)
            self.assertTrue(access["accessible"])

            action = {
                "type": "idle",
                "duration": 1.0,
                "priority": 1,
                "progress": 0.0,
                "artifact_action": ExecutableActionType.CONSULT_TEAM_ARTIFACT.value,
                "decision_action": ExecutableActionType.CONSULT_TEAM_ARTIFACT.value,
                "execution_stage": None,
            }
            agent.position = target
            agent.active_actions = [action]

            agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
            self.assertIn(action.get("execution_stage"), {"selected", "mutation_execution_started", "mutation_execution_succeeded"})
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
