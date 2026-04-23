import unittest
from unittest import mock

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class ConstructionLifecycleOrderingTests(unittest.TestCase):
    def _prime_validation_knowledge(self, sim, agent, project_id):
        project = sim.environment.construction.projects[project_id]
        expected = list(project.get("expected_rules") or [])
        for rule_id in expected:
            if rule_id not in agent.mental_model["knowledge"].rules:
                agent.mental_model["knowledge"].rules.append(rule_id)

    def test_material_completion_alone_is_not_validation_ready(self):
        sim = SimulationState(phases=[])
        project_id = sim.environment.construction.resolve_project_id("Build_Table_B", create_if_missing=True)
        project = sim.environment.construction.projects[project_id]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)

        self.assertTrue(project["resource_complete"])
        self.assertFalse(project["structurally_complete"])
        self.assertNotEqual(project["status"], "ready_for_validation")
        sim.stop()

    def test_validate_action_blocked_until_physical_build_progress(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = sim.environment.construction.resolve_project_id("Build_Table_B", create_if_missing=True)
        project = sim.environment.construction.projects[project_id]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        self._prime_validation_knowledge(sim, agent, project_id)

        decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
        blockers, resolved_project_id = agent._construction_action_blockers(
            decision,
            {"type": "idle", "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value, "project_id": project_id},
            sim.environment,
            sim_state=sim,
        )
        self.assertEqual(resolved_project_id, project_id)
        self.assertIn("physical_build_not_started", blockers)
        sim.stop()

    def test_mismatch_detection_suppressed_when_materials_only(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = sim.environment.construction.resolve_project_id("Build_Table_B", create_if_missing=True)
        project = sim.environment.construction.projects[project_id]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        self._prime_validation_knowledge(sim, agent, project_id)

        with mock.patch("modules.agent.random.random", return_value=0.0):
            agent.compare_and_repair_construction(sim.environment.construction, sim_state=sim)

        event_types = [e.get("event_type") for e in sim.logger.get_recent_events(200)]
        self.assertIn("mismatch_detection_skipped_not_ready", event_types)
        self.assertNotIn("construction_mismatch_detected", event_types)
        sim.stop()

    def test_build_progress_then_validation_readiness_can_activate(self):
        sim = SimulationState(phases=[])
        project_id = sim.environment.construction.resolve_project_id("Build_Table_B", create_if_missing=True)
        project = sim.environment.construction.projects[project_id]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)

        ok, reason, _ = sim.environment.construction.execute_build_step(project_id, actor="Architect", sim_time=sim.time)
        self.assertTrue(ok, reason)
        self.assertTrue(any(step.get("completed") for step in project.get("build_steps", [])))

        for entry_type in ("claim", "evidence", "design_note"):
            sim.environment.construction.record_project_epistemic_externalization(
                project_id,
                entry_type=entry_type,
                note=entry_type,
                actor="Architect",
                sim_time=sim.time,
            )

        # Project may still be in-progress after first step, but once fully built it can become ready.
        while not project.get("structurally_complete"):
            sim.environment.construction.execute_build_step(project_id, actor="Architect", sim_time=sim.time)
        sim.environment.construction.update()
        self.assertEqual(project.get("status"), "ready_for_validation")
        sim.stop()


if __name__ == "__main__":
    unittest.main()
