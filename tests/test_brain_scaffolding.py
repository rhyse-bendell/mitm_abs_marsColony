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
from modules.brain_provider import BrainBackendConfig, RuleBrain, create_brain_provider
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


class TestDecisionRoutingAndPlanning(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_authoritative_brain_route_is_used_in_simulation_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            calls = []

            def wrapped(sim_state_arg, environment_arg):
                reason = original(sim_state_arg, environment_arg)
                calls.append(reason)
                return reason

            original = agent._plan_trigger_reason
            agent._plan_trigger_reason = wrapped

            sim.update(0.2)

            self.assertTrue(calls)

    def test_cached_plan_prevents_replanning_every_tick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            query_calls = []
            original = agent._build_rule_based_brain_decision

            def wrapped(sim_state, trigger_reason):
                query_calls.append(trigger_reason)
                return original(sim_state, trigger_reason)

            agent._build_rule_based_brain_decision = wrapped

            sim.update(0.2)
            sim.update(0.2)

            self.assertEqual(len(query_calls), 1)

    def test_trigger_policy_replans_on_new_dik(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]

            sim.update(0.2)
            first_plan_id = agent.current_plan.plan_id

            agent.mental_model["knowledge"].rules.append("synthetic_rule_for_replan")

            for _ in range(8):
                sim.update(0.2)
                if agent.current_plan.plan_id != first_plan_id:
                    break

            self.assertNotEqual(agent.current_plan.plan_id, first_plan_id)

    def test_plan_persistence_carries_action_across_ticks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]

            sim.update(0.2)
            plan_id = agent.current_plan.plan_id
            remaining_before = agent.current_plan.remaining_executions

            for _ in range(8):
                sim.update(0.2)
                if agent.current_plan.plan_id != plan_id or agent.current_plan.remaining_executions < remaining_before:
                    break

            self.assertEqual(agent.current_plan.plan_id, plan_id)
            self.assertLess(agent.current_plan.remaining_executions, remaining_before)

    def test_memory_compaction_exposes_three_history_bands(self):
        agent = Agent(name="Architect", role="Architect")
        for i in range(20):
            agent.activity_log.append(f"event-{i}")

            
        bands = agent.history_bands()

        self.assertIn("current_state_summary", bands)
        self.assertIn("near_preceding_events", bands)
        self.assertIn("recent_history_summary", bands)
        self.assertLessEqual(len(bands["near_preceding_events"]), 8)


class TestBackendConfiguration(unittest.TestCase):
    def test_rule_brain_is_default_backend(self):
        provider = create_brain_provider(BrainBackendConfig())
        self.assertIsInstance(provider, RuleBrain)

    def test_simulation_uses_configurable_backend_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="local_stub")
            self.assertEqual(sim.brain_backend_config.backend, "local_stub")
            self.assertEqual(sim.brain_provider.__class__.__name__, "LocalLLMBrainStub")



if __name__ == "__main__":
    unittest.main()
