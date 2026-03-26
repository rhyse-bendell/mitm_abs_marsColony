import unittest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState
from modules.task_model import load_task_model


class ConstructionEpistemicAuthorityTests(unittest.TestCase):
    def _prime_build_readiness(self, sim, agent):
        team_packet = sim.environment.knowledge_packets["Team_Info"]
        agent.mental_model["information"].add(team_packet["information"][0])
        agent.mental_model["information"].add(team_packet["information"][1])
        agent.mental_model["knowledge"].rules.append("R_HOUSE_VALIDITY")
        agent.source_inspection_state["Team_Info"] = "inspected"

    def test_start_construction_blocked_when_epistemic_prereqs_missing(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        decision = BrainDecision(
            selected_action=ExecutableActionType.START_CONSTRUCTION,
            target_id="Build_Table_B",
            confidence=0.9,
        )

        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)

        self.assertEqual(translated[0]["type"], "idle")
        self.assertEqual(translated[0].get("decision_action"), ExecutableActionType.WAIT.value)
        sim.stop()

    def test_start_construction_allowed_when_grounded(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)

        decision = BrainDecision(
            selected_action=ExecutableActionType.START_CONSTRUCTION,
            target_id="Build_Table_B",
            confidence=0.9,
        )
        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)

        self.assertEqual(translated[0]["type"], "construct")
        self.assertEqual(translated[0].get("project_id"), "Build_Table_B")
        sim.stop()

    def test_resource_completion_not_equal_validated_completion(self):
        model = load_task_model("mars_colony")
        project = model.construction_templates["Build_Table_B"]
        sim = SimulationState(phases=[])

        for _ in range(project.required_resources["bricks"]):
            sim.environment.construction.deliver_resource("Build_Table_B", "bricks", quantity=1)

        p = sim.environment.construction.projects["Build_Table_B"]
        self.assertTrue(p["resource_complete"])
        self.assertFalse(p["validated_complete"])
        self.assertNotEqual(p["status"], "complete")
        sim.stop()

    def test_repair_then_validate_can_complete_project(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)

        project = sim.environment.construction.projects["Build_Table_B"]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource("Build_Table_B", "bricks", quantity=required)
        project["correct"] = False

        agent.activity_log.append("Mismatch with construction: reevaluating knowledge")
        repair = BrainDecision(
            selected_action=ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
            target_id="Build_Table_B",
            confidence=0.8,
        )
        repair_action = agent._translate_brain_decision_to_legacy_action(repair, sim.environment, sim_state=sim)[0]
        self.assertEqual(repair_action["type"], "construct")

        agent.inventory_resources["bricks"] = 1
        agent.position = sim.environment.get_interaction_target_position("Build_Table_B", from_position=agent.position)
        agent.active_actions = [{**repair_action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertTrue(project["correct"])

        validate = BrainDecision(
            selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION,
            target_id="Build_Table_B",
            confidence=0.8,
        )
        validate_action = agent._translate_brain_decision_to_legacy_action(validate, sim.environment, sim_state=sim)[0]
        self.assertEqual(validate_action["type"], "idle")

        agent.position = sim.environment.get_interaction_target_position("Build_Table_B", from_position=agent.position)
        agent.active_actions = [{**validate_action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        self.assertTrue(project["validated_complete"])
        self.assertEqual(project["status"], "complete")
        sim.stop()

    def test_start_construction_auto_handoffs_to_logistics_when_resources_missing(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        project = sim.environment.construction.projects["Build_Table_B"]
        before = int(project["delivered_resources"]["bricks"])

        decision = BrainDecision(
            selected_action=ExecutableActionType.START_CONSTRUCTION,
            target_id="Build_Table_B",
            confidence=0.9,
        )
        action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        agent.inventory_resources["bricks"] = 0
        agent.active_actions = [{**action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        after = int(project["delivered_resources"]["bricks"])
        self.assertEqual(after, before)
        sim.stop()

    def test_transport_does_not_false_progress_when_resources_already_satisfied(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project = sim.environment.construction.projects["Build_Table_B"]
        required = int(project["required_resources"]["bricks"])
        project["delivered_resources"]["bricks"] = required
        sim.environment.construction.update()
        before = int(project["delivered_resources"]["bricks"])

        decision = BrainDecision(
            selected_action=ExecutableActionType.TRANSPORT_RESOURCES,
            target_id="Build_Table_B",
            confidence=0.9,
        )
        action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        agent.active_actions = [{**action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        after = int(project["delivered_resources"]["bricks"])

        self.assertEqual(after, before)
        sim.stop()

    def test_transport_requires_pickup_and_dropoff_legality(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        before = int(project["delivered_resources"]["bricks"])
        transport = {
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
        }
        agent.position = (8.0, 6.6)  # Team_Info region; not pickup and not build table
        agent.active_actions = [transport]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertEqual(before, int(project["delivered_resources"]["bricks"]))
        self.assertEqual(agent.transport_state.get("stage"), "pickup")
        sim.stop()

    def test_validation_requires_location_and_status(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_build_readiness(sim, agent)
        project_id = "Build_Table_B"
        validate = {
            "type": "idle",
            "duration": 1.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.VALIDATE_CONSTRUCTION.value,
        }
        # Wrong location and status: should not validate.
        agent.position = (8.0, 6.6)
        agent.active_actions = [validate]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertFalse(sim.environment.construction.projects[project_id]["validated_complete"])
        self.assertTrue(any(e.get("event_type") == "construction_validation_blocked" for e in sim.logger.recent_events))
        sim.stop()

    def test_inspect_context_cannot_shortcut_delivery(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        before = int(project["delivered_resources"]["bricks"])
        # Simulate stale inspect context while a transport action exists.
        agent.current_inspect_target_id = "Team_Info"
        agent.position = (8.0, 6.6)
        agent.active_actions = [{
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "project_id": project_id,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
        }]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertEqual(before, int(project["delivered_resources"]["bricks"]))
        self.assertFalse(any(e.get("event_type") == "construction_resource_delivered" for e in sim.logger.recent_events))
        sim.stop()

    def test_construction_expected_rules_normalized_to_canonical_ids(self):
        model = load_task_model("mars_colony")
        for template in model.construction_templates.values():
            self.assertTrue(template.expected_rules)
            self.assertTrue(all(rule.startswith("R_") for rule in template.expected_rules))


if __name__ == "__main__":
    unittest.main()
