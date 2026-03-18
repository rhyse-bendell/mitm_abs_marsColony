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

        agent.active_actions = [{**validate_action, "progress": 0.0}]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)

        self.assertTrue(project["validated_complete"])
        self.assertEqual(project["status"], "complete")
        sim.stop()

    def test_construction_expected_rules_normalized_to_canonical_ids(self):
        model = load_task_model("mars_colony")
        for template in model.construction_templates.values():
            self.assertTrue(template.expected_rules)
            self.assertTrue(all(rule.startswith("R_") for rule in template.expected_rules))


if __name__ == "__main__":
    unittest.main()
