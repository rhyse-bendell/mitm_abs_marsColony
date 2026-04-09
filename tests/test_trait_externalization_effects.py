import random
import tempfile
import unittest
from unittest.mock import patch

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_context import BrainContextBuilder
from modules.brain_provider import RuleBrain
from modules.simulation import SimulationState


class TestTraitDrivenBehavior(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def _make_sim(self):
        return tempfile.TemporaryDirectory()

    def test_communication_propensity_changes_externalization_tendency(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            context = BrainContextBuilder().build(sim, agent)
            agent.communication_propensity = 0.9
            with patch("modules.agent.random.random", return_value=0.1):
                high = agent._apply_trait_bias_to_decision(
                    BrainDecision(selected_action=ExecutableActionType.WAIT),
                    context,
                    sim,
                    "new_dik_acquired",
                )
            agent.communication_propensity = 0.2
            with patch("modules.agent.random.random", return_value=0.1):
                low = agent._apply_trait_bias_to_decision(
                    BrainDecision(selected_action=ExecutableActionType.WAIT),
                    context,
                    sim,
                    "new_dik_acquired",
                )
            self.assertEqual(high.selected_action, ExecutableActionType.EXTERNALIZE_PLAN)
            self.assertNotEqual(low.selected_action, ExecutableActionType.EXTERNALIZE_PLAN)

    def test_help_tendency_changes_response_under_help_context(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            agent.known_gaps.add("need_rule_clarification")
            context = BrainContextBuilder().build(sim, agent)
            agent.help_tendency = 0.9
            with patch("modules.agent.random.random", return_value=0.2):
                high = agent._apply_trait_bias_to_decision(
                    BrainDecision(selected_action=ExecutableActionType.WAIT),
                    context,
                    sim,
                    "help_context",
                )
            agent.help_tendency = 0.1
            with patch("modules.agent.random.random", return_value=0.2):
                low = agent._apply_trait_bias_to_decision(
                    BrainDecision(selected_action=ExecutableActionType.WAIT),
                    context,
                    sim,
                    "help_context",
                )
            self.assertEqual(high.selected_action, ExecutableActionType.REQUEST_ASSISTANCE)
            self.assertNotEqual(low.selected_action, ExecutableActionType.REQUEST_ASSISTANCE)

    def test_build_speed_scales_macro_action_durations(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]

            decision = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES)

            agent.build_speed = 1.0
            fast_action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment)[0]

            agent.build_speed = 0.0
            slow_action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment)[0]

            self.assertLess(fast_action["duration"], slow_action["duration"])

    def test_rule_accuracy_changes_construction_externalization_fidelity(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            agent.rule_accuracy = 0.95
            high_fidelity = (
                agent._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5)
                + agent._trait_value("rule_accuracy")
            ) / 2.0
            agent.rule_accuracy = 0.2
            low_fidelity = (
                agent._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5)
                + agent._trait_value("rule_accuracy")
            ) / 2.0
            self.assertGreater(high_fidelity, low_fidelity)

    def test_goal_alignment_influences_shared_artifact_use(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            context = BrainContextBuilder().build(sim, agent)
            context.team_state["plan_readiness"] = "validated_shared_plan"
            agent.goal_alignment = 0.9
            with patch("modules.agent.random.random", return_value=0.1):
                high = agent._apply_trait_bias_to_decision(
                    BrainDecision(selected_action=ExecutableActionType.WAIT),
                    context,
                    sim,
                    "no_active_plan",
                )
            agent.goal_alignment = 0.2
            with patch("modules.agent.random.random", return_value=0.1):
                low = agent._apply_trait_bias_to_decision(
                    BrainDecision(selected_action=ExecutableActionType.WAIT),
                    context,
                    sim,
                    "no_active_plan",
                )
            self.assertEqual(high.selected_action, ExecutableActionType.CONSULT_TEAM_ARTIFACT)
            self.assertNotEqual(low.selected_action, ExecutableActionType.CONSULT_TEAM_ARTIFACT)

    def test_constructions_are_externalized_as_team_artifacts(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            sim.update(0.2)
            artifact_ids = [aid for aid in sim.team_knowledge_manager.artifacts if aid.startswith("construction:")]
            self.assertTrue(artifact_ids)
            artifact = sim.team_knowledge_manager.artifacts[artifact_ids[0]]
            self.assertTrue(artifact.artifact_type.startswith("construction_"))

    def test_headless_simulation_still_runs(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            sim.update(0.5)
            self.assertGreater(sim.time, 0.0)

    def test_mismatch_detection_sensitivity_hook_changes_detection_likelihood(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            env = sim.environment
            project = env.construction.projects["Build_Table_B"]
            project["in_progress"] = True
            project["required_resources"] = {"bricks": 10}
            project["delivered_resources"] = {"bricks": 8}
            project["expected_rules"] = ["rule:needs_other_rule"]
            agent.mental_model["knowledge"].rules = ["rule:my_rule"]
            agent.rule_accuracy = 0.2

            agent.hook_effects[("validation_check", "detect_mismatch", "sensitivity")] = 0.1
            with patch("modules.agent.random.random", return_value=0.6):
                agent.compare_and_repair_construction(env.construction, sim_state=sim)
            detected_low = any("Disagrees with approach" in entry for entry in agent.activity_log)

            agent.activity_log.clear()
            agent.hook_effects[("validation_check", "detect_mismatch", "sensitivity")] = 0.95
            agent.rule_accuracy = 0.2
            with patch("modules.agent.random.random", return_value=0.6):
                agent.compare_and_repair_construction(env.construction, sim_state=sim)
            detected_high = any("Disagrees with approach" in entry for entry in agent.activity_log)

            self.assertFalse(detected_low)
            self.assertTrue(detected_high)


if __name__ == "__main__":
    unittest.main()
