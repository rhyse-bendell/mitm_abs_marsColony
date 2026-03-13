import random
import tempfile
import unittest

from modules.action_schema import (
    BrainDecision,
    ExecutableActionType,
    validate_brain_decision,
)
from modules.agent import Agent
from modules.brain_context import BrainContextBuilder
from modules.brain_provider import RuleBrain
from modules.environment import Environment
from modules.simulation import SimulationState
from modules.team_knowledge import TeamKnowledgeManager


class TestActionSchema(unittest.TestCase):
    def test_action_families_present(self):
        required = {
            ExecutableActionType.MOVE_TO_TARGET,
            ExecutableActionType.INSPECT_INFORMATION_SOURCE,
            ExecutableActionType.COMMUNICATE,
            ExecutableActionType.REQUEST_ASSISTANCE,
            ExecutableActionType.MEETING,
            ExecutableActionType.EXTERNALIZE_PLAN,
            ExecutableActionType.CONSULT_TEAM_ARTIFACT,
            ExecutableActionType.TRANSPORT_RESOURCES,
            ExecutableActionType.START_CONSTRUCTION,
            ExecutableActionType.CONTINUE_CONSTRUCTION,
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
            ExecutableActionType.VALIDATE_CONSTRUCTION,
            ExecutableActionType.OBSERVE_ENVIRONMENT,
            ExecutableActionType.REASSESS_PLAN,
            ExecutableActionType.WAIT,
        }
        self.assertTrue(required.issubset(set(ExecutableActionType)))


class TestBrainContextAndDecision(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_brain_context_builds_in_headless_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = sim.brain_context_builder.build(sim, agent)
            self.assertIn("mission", packet.static_task_context)
            self.assertIn("sim_time", packet.world_snapshot)
            self.assertGreater(len(packet.action_affordances), 0)

    def test_brain_decision_validation_rejects_illegal(self):
        decision = BrainDecision(selected_action=ExecutableActionType.START_CONSTRUCTION, confidence=0.9)
        errors = validate_brain_decision(decision, legal_actions=[ExecutableActionType.WAIT])
        self.assertTrue(errors)

    def test_rule_brain_returns_structured_valid_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = BrainContextBuilder().build(sim, agent)
            decision = RuleBrain().decide(packet)
            legal = [ExecutableActionType(a["action_type"]) for a in packet.action_affordances]
            errors = validate_brain_decision(decision, legal)
            self.assertEqual(errors, [])
            self.assertIsInstance(decision.reason_summary, str)


class TestTeamKnowledgeManagerAndIntegration(unittest.TestCase):
    def test_team_knowledge_manager_store_and_retrieve_artifact(self):
        manager = TeamKnowledgeManager()
        manager.externalize_artifact(
            artifact_id="plan-alpha",
            artifact_type="plan",
            summary="Build greenhouse first",
            content={"steps": ["gather", "assemble"]},
            author="Architect",
            sim_time=12.0,
        )
        artifact = manager.get_artifact("plan-alpha")
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.author, "Architect")

    def test_headless_simulation_runs_with_rule_brain_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            sim.update(0.5)
            self.assertGreater(sim.time, 0.0)
            self.assertIsNotNone(sim.brain_provider)

    def test_transport_resources_is_single_macro_action(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.get_spawn_point("Engineer"))
        decision = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, assumptions=["duration_s=30"])
        actions = agent._translate_brain_decision_to_legacy_action(decision, env)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "transport_resources")
        self.assertGreaterEqual(actions[0]["duration"], 25.0)


if __name__ == "__main__":
    unittest.main()
