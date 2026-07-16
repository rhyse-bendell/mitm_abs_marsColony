import unittest
from unittest.mock import patch

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class TestValidationTargetBinding(unittest.TestCase):
    def _project_id(self, sim):
        cm = sim.environment.construction
        project_id, _ = cm.create_project("site_b", structure_type="connector", project_id_override="site_b_food_connector_001")
        return project_id

    def test_validation_translation_binds_project_target(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            project_id = self._project_id(sim)
            project = sim.environment.construction.projects[project_id]
            project["delivered_resources"]["bricks"] = project["required_resources"]["bricks"]
            project["resource_complete"] = True
            project["structurally_complete"] = True
            project["functional_support_complete"] = True
            project["support_requirements"] = {}
            project["epistemic_workspace"]["entries"] = [
                {"entry_type": "claim"},
                {"entry_type": "evidence"},
                {"entry_type": "design_note"},
            ]
            decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
            with patch.object(agent, "_construction_action_blockers", return_value=([], project_id)):
                actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
            action = actions[0]
            self.assertEqual(action.get("project_id"), project_id)
            self.assertEqual(action.get("target_id"), project_id)
            self.assertEqual(action.get("target_kind"), "construction_project")
            self.assertNotEqual(action.get("target_id"), "whiteboard")
            self.assertEqual(tuple(action.get("target") or ()), tuple(sim.environment.get_interaction_target_position(project_id, from_position=agent.position) or ()))
        finally:
            sim.stop()

    def test_stale_whiteboard_target_is_repaired(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            project_id = self._project_id(sim)
            action = {
                "type": "idle",
                "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
                "project_id": project_id,
                "target_id": "whiteboard",
                "target_kind": "artifact",
                "target": sim.environment.get_interaction_target_position("whiteboard", from_position=agent.position),
            }
            repaired = agent._validate_and_repair_validation_target_binding(action, sim.environment, sim_state=sim)
            self.assertEqual(repaired.get("target_id"), project_id)
            self.assertEqual(repaired.get("target_kind"), "construction_project")
            self.assertNotEqual(repaired.get("target_id"), "whiteboard")
        finally:
            sim.stop()

    def test_validation_en_route_does_not_complete_idle(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            project_id = self._project_id(sim)
            target = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
            agent.position = (0.0, 0.0)
            action = {
                "type": "idle",
                "duration": 1.0,
                "priority": 1,
                "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
                "project_id": project_id,
                "target_id": "whiteboard",
                "target_kind": "artifact",
                "target": target,
                "progress": 0.0,
            }
            agent.active_actions = [action]
            agent._apply_externalization_and_construction_effects(sim.environment, sim, 0.1)
            self.assertIn(action.get("execution_stage"), {"en_route", "selected"})
            self.assertEqual(action.get("project_id"), project_id)
            self.assertEqual(action.get("target_id"), project_id)
            self.assertFalse(sim.environment.construction.projects[project_id].get("validated_complete", False))
        finally:
            sim.stop()

    def test_validation_occurs_after_arrival(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            project_id = self._project_id(sim)
            cm = sim.environment.construction
            project = cm.projects[project_id]
            project["status"] = "ready_for_validation"
            req = int((project.get("required_resources") or {}).get("bricks", 0) or 0)
            if req > 0:
                cm.deliver_resource(project_id, "bricks", quantity=req)
            remaining_steps = sum(1 for step in project.get("build_steps", []) if not step.get("completed"))
            for _ in range(remaining_steps):
                cm.execute_build_step(project_id, actor=agent.name, sim_time=sim.time)
            for entry_type in ("claim", "evidence", "design_note"):
                cm.record_project_epistemic_externalization(
                    project_id,
                    entry_type=entry_type,
                    note=f"{entry_type} for validation",
                    references=["R_HOUSE_VALIDITY"],
                    actor=agent.name,
                    sim_time=sim.time,
                )
            project["support_requirements"] = {}
            project["support_counts"] = {}
            project["support_status"] = {}
            cm.recompute_support_status(project_id)
            project["correct"] = True
            agent.position = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
            action = {
                "type": "idle",
                "duration": 1.0,
                "priority": 1,
                "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
                "project_id": project_id,
                "target_id": project_id,
                "target_kind": "construction_project",
                "target": agent.position,
                "progress": 0.0,
            }
            agent.active_actions = [action]
            agent._apply_externalization_and_construction_effects(sim.environment, sim, 0.1)
            events = [e.get("event_type") for e in sim.logger.get_recent_events(80)]
            self.assertIn("construction_validation_attempted", events)
            self.assertIn(action.get("execution_stage"), {"mutation_execution_started", "mutation_execution_succeeded", "arrived"})
        finally:
            sim.stop()

    def test_validation_persists_across_ticks_until_arrival_then_validates(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            project_id = self._project_id(sim)
            cm = sim.environment.construction
            project = cm.projects[project_id]
            project["status"] = "ready_for_validation"
            req = int((project.get("required_resources") or {}).get("bricks", 0) or 0)
            if req > 0:
                cm.deliver_resource(project_id, "bricks", quantity=req)
            for entry_type in ("claim", "evidence", "design_note"):
                cm.record_project_epistemic_externalization(
                    project_id,
                    entry_type=entry_type,
                    note=f"{entry_type} for validation",
                    references=["R_HOUSE_VALIDITY"],
                    actor=agent.name,
                    sim_time=sim.time,
                )
            project["support_requirements"] = {}
            project["support_counts"] = {}
            project["support_status"] = {}
            cm.recompute_support_status(project_id)
            project["correct"] = True

            agent.position = (0.0, 0.0)
            for expected_rule in list(project.get("expected_rules") or []):
                if expected_rule not in agent.mental_model["knowledge"].rules:
                    agent.mental_model["knowledge"].rules.append(expected_rule)
            action = {
                "type": "idle",
                "duration": 1.0,
                "priority": 1,
                "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
                "project_id": project_id,
                "target_id": "whiteboard",
                "target_kind": "artifact",
                "target": sim.environment.get_interaction_target_position("whiteboard", from_position=agent.position),
                "progress": 0.0,
            }
            agent.active_actions = [action]

            for _ in range(3):
                agent._apply_externalization_and_construction_effects(sim.environment, sim, 0.1)
                agent._advance_active_actions(0.6, sim_state=sim)
                self.assertIn(action, agent.active_actions)
                self.assertEqual(action.get("project_id"), project_id)
                self.assertEqual(action.get("target_id"), project_id)
                self.assertEqual(action.get("target_kind"), "construction_project")
                self.assertEqual(action.get("validation_target_project_id"), project_id)
                expected_target = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
                self.assertEqual(tuple(action.get("target") or ()), tuple(expected_target or ()))
                self.assertLess(action.get("progress", 0.0), action.get("duration", 1.0))

            while action in agent.active_actions:
                target = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
                dx = target[0] - agent.position[0]
                dy = target[1] - agent.position[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > 0:
                    step = min(1.0, dist)
                    agent.position = (agent.position[0] + (dx / dist) * step, agent.position[1] + (dy / dist) * step)
                agent._apply_externalization_and_construction_effects(sim.environment, sim, 0.1)
                agent._advance_active_actions(0.6, sim_state=sim)

            events = [e.get("event_type") for e in sim.logger.get_recent_events(300)]
            self.assertIn("validation_arrival_pending", events)
            self.assertIn("validation_arrival_confirmed", events)
            self.assertIn("construction_validation_attempted", events)
            readiness = cm.evaluate_project_validation_readiness(
                project_id,
                actor=agent.name,
                agent_supported_rules=list(agent.mental_model["knowledge"].rules),
            )
            if readiness.get("validation_ready"):
                self.assertTrue(cm.projects[project_id].get("validated_complete", False))
        finally:
            sim.stop()

    def test_non_validation_artifact_actions_are_unaffected(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            action = {
                "type": "idle",
                "decision_action": ExecutableActionType.CONSULT_TEAM_ARTIFACT.value,
                "target_id": "whiteboard",
                "target_kind": "artifact",
                "target": sim.environment.get_interaction_target_position("whiteboard", from_position=agent.position),
            }
            untouched = agent._validate_and_repair_validation_target_binding(action, sim.environment, sim_state=sim)
            self.assertEqual(untouched.get("target_id"), "whiteboard")
            self.assertEqual(untouched.get("target_kind"), "artifact")
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
