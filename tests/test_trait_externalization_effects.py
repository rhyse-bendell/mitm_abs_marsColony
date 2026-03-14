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
            agent.mental_model["knowledge"].add_rule("rule:a", ["I1"])

            agent.communication_propensity = 0.9
            packet_high = BrainContextBuilder().build(sim, agent)
            high_decision = RuleBrain().decide(packet_high)

            agent.communication_propensity = 0.2
            packet_low = BrainContextBuilder().build(sim, agent)
            low_decision = RuleBrain().decide(packet_low)

            self.assertEqual(high_decision.selected_action, ExecutableActionType.EXTERNALIZE_PLAN)
            self.assertNotEqual(low_decision.selected_action, ExecutableActionType.EXTERNALIZE_PLAN)

    def test_help_tendency_changes_response_under_help_context(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            agent.known_gaps.add("need_rule_clarification")

            agent.help_tendency = 0.9
            packet_high = BrainContextBuilder().build(sim, agent)
            high_decision = RuleBrain().decide(packet_high)

            agent.help_tendency = 0.1
            packet_low = BrainContextBuilder().build(sim, agent)
            low_decision = RuleBrain().decide(packet_low)

            self.assertEqual(high_decision.selected_action, ExecutableActionType.REQUEST_ASSISTANCE)
            self.assertNotEqual(low_decision.selected_action, ExecutableActionType.REQUEST_ASSISTANCE)

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
            env = sim.environment
            project = env.construction.projects["Build_Table_B"]

            agent.active_actions = [{"type": "construct", "progress": 0, "duration": 1.0, "priority": 1}]
            agent.rule_accuracy = 0.95
            with patch("modules.agent.random.random", return_value=0.8):
                agent._apply_externalization_and_construction_effects(env, sim, dt=0.1)
            self.assertTrue(project["correct"])

            project["correct"] = True
            agent.active_actions = [{"type": "construct", "progress": 0, "duration": 1.0, "priority": 1}]
            agent.rule_accuracy = 0.2
            with patch("modules.agent.random.random", return_value=0.8):
                agent._apply_externalization_and_construction_effects(env, sim, dt=0.1)
            self.assertFalse(project["correct"])

    def test_goal_alignment_influences_shared_artifact_use(self):
        with self._make_sim() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            sim.team_knowledge_manager.externalize_artifact(
                artifact_id="whiteboard:validated",
                artifact_type="whiteboard_plan",
                summary="validated",
                content={"rules": ["r1"]},
                author="Architect",
                sim_time=0.0,
                validation_state="validated",
            )

            agent.goal_alignment = 0.9
            high_packet = BrainContextBuilder().build(sim, agent)
            high_decision = RuleBrain().decide(high_packet)

            agent.goal_alignment = 0.2
            low_packet = BrainContextBuilder().build(sim, agent)
            low_decision = RuleBrain().decide(low_packet)

            self.assertEqual(high_decision.selected_action, ExecutableActionType.CONSULT_TEAM_ARTIFACT)
            self.assertNotEqual(low_decision.selected_action, ExecutableActionType.CONSULT_TEAM_ARTIFACT)

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


if __name__ == "__main__":
    unittest.main()
