import tempfile
import unittest

from modules.simulation import SimulationState


class TestTeamKnowledgePropagation(unittest.TestCase):
    def test_team_validated_knowledge_supports_derivation_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            task = sim.task_model

            derivation = next((d for d in task.derivations.values() if d.enabled and d.required_inputs and d.output_type == "knowledge"), None)
            self.assertIsNotNone(derivation)
            output_rule = derivation.output_element_id

            for req in derivation.required_inputs:
                elem = task.dik_elements.get(req)
                self.assertIsNotNone(elem)
                sim.team_knowledge_manager.add_validated_knowledge(
                    f"{elem.element_type}:{req}",
                    f"seed:{req}",
                    author="unit_test",
                    sim_time=sim.time,
                )

            self.assertNotIn(output_rule, agent.mental_model["knowledge"].rules)
            agent._apply_task_derivations(sim_state=sim)
            self.assertIn(output_rule, agent.mental_model["knowledge"].rules)

            blockers = agent._build_readiness_blockers(sim.environment, sim_state=sim)
            self.assertNotIn("insufficient_information_inspection", blockers)
            self.assertNotIn("insufficient_rule_knowledge", blockers)
            sim.stop()


if __name__ == "__main__":
    unittest.main()
