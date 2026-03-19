import json
import random
import tempfile
import unittest
from unittest.mock import patch

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
from urllib import error

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
            sim = SimulationState(phases=[], project_root=tmpdir, planner_config={"planner_interval_steps": 50, "planner_interval_time": 999.0, "planner_trigger_mask": []})
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

    def test_inspect_affordances_include_explicit_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = BrainContextBuilder().build(sim, agent)
            inspect_affordances = [
                a
                for a in packet.action_affordances
                if a["action_type"] == ExecutableActionType.INSPECT_INFORMATION_SOURCE.value
            ]
            self.assertTrue(inspect_affordances)
            self.assertTrue(all(a.get("target_id") for a in inspect_affordances))
            self.assertTrue(all(a.get("target_zone") for a in inspect_affordances))

    def test_rule_brain_prioritizes_productive_build_after_readiness_unlock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = BrainContextBuilder().build(sim, agent)

            packet.world_snapshot["phase_profile"]["stage"] = "execution"
            packet.individual_cognitive_state["known_gaps"] = ["need_help"]
            packet.individual_cognitive_state["traits"]["help_tendency"] = 0.95
            packet.individual_cognitive_state["build_readiness"]["ready_for_build"] = True
            packet.individual_cognitive_state["build_readiness"]["status"] = "plausible"
            packet.world_snapshot["built_state"] = [
                {
                    "structure_id": "Build_Table_A",
                    "state": "in_progress",
                    "progress": 0.0,
                }
            ]
            packet.action_affordances = [
                {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "utility": 0.95, "target_id": "nearby_agent"},
                {"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "utility": 0.5, "target_id": "resource_zone_to_work_zone"},
                {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.4, "target_id": "Build_Table_A"},
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.3, "target_id": "Team_Info"},
            ]

            decision = RuleBrain().decide(packet)
            self.assertIn(
                decision.selected_action,
                {
                    ExecutableActionType.TRANSPORT_RESOURCES,
                    ExecutableActionType.START_CONSTRUCTION,
                    ExecutableActionType.CONTINUE_CONSTRUCTION,
                },
            )
            self.assertNotEqual(decision.selected_action, ExecutableActionType.REQUEST_ASSISTANCE)

    def test_rule_brain_keeps_assistance_when_no_productive_build_affordance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = BrainContextBuilder().build(sim, agent)

            packet.world_snapshot["phase_profile"]["stage"] = "execution"
            packet.individual_cognitive_state["known_gaps"] = ["need_help"]
            packet.individual_cognitive_state["traits"]["help_tendency"] = 0.95
            packet.individual_cognitive_state["build_readiness"]["ready_for_build"] = True
            packet.individual_cognitive_state["build_readiness"]["status"] = "plausible"
            packet.world_snapshot["built_state"] = [
                {
                    "structure_id": "Build_Table_A",
                    "state": "in_progress",
                    "progress": 0.0,
                }
            ]
            packet.action_affordances = [
                {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "utility": 0.95, "target_id": "nearby_agent"},
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.6, "target_id": "Team_Info"},
                {"action_type": ExecutableActionType.WAIT.value, "utility": 0.1},
            ]

            decision = RuleBrain().decide(packet)
            self.assertEqual(decision.selected_action, ExecutableActionType.REQUEST_ASSISTANCE)

    def test_rule_brain_pivots_to_start_when_ready_with_absent_projects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = BrainContextBuilder().build(sim, agent)

            packet.world_snapshot["phase_profile"]["stage"] = "execution"
            packet.individual_cognitive_state["known_gaps"] = ["need_help"]
            packet.individual_cognitive_state["traits"]["help_tendency"] = 0.95
            packet.individual_cognitive_state["build_readiness"]["ready_for_build"] = True
            packet.individual_cognitive_state["build_readiness"]["status"] = "plausible"
            packet.individual_cognitive_state["seconds_since_dik_change"] = 9.0
            packet.world_snapshot["built_state"] = [
                {
                    "structure_id": "Build_Table_A",
                    "state": "absent",
                    "progress": 0.0,
                }
            ]
            packet.action_affordances = [
                {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "utility": 0.95, "target_id": "nearby_agent"},
                {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.4, "target_id": "Build_Table_A"},
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.6, "target_id": "Team_Info"},
            ]

            decision = RuleBrain().decide(packet)
            self.assertEqual(decision.selected_action, ExecutableActionType.START_CONSTRUCTION)


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


    def test_planner_cadence_interval_prevents_per_tick_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                planner_config={
                    "planner_interval_steps": 50,
                    "planner_interval_time": 999.0,
                    "planner_trigger_mask": [],
                },
            )
            agent = sim.agents[0]
            query_calls = []
            original = agent._build_rule_based_brain_decision

            def wrapped(sim_state, trigger_reason):
                query_calls.append(trigger_reason)
                return original(sim_state, trigger_reason)

            agent._build_rule_based_brain_decision = wrapped
            sim.update(0.2)
            for _ in range(6):
                sim.update(0.2)

            self.assertEqual(len(query_calls), 1)

    def test_trigger_mask_replans_when_enabled_trigger_fires(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                planner_config={
                    "planner_interval_steps": 50,
                    "planner_interval_time": 999.0,
                    "planner_trigger_mask": ["new_dik_acquired"],
                },
            )
            agent = sim.agents[0]
            sim.update(0.2)
            first_plan_id = agent.current_plan.plan_id

            agent.mental_model["knowledge"].rules.append("synthetic_rule_for_replan")
            for _ in range(8):
                sim.update(0.2)
                if agent.current_plan.plan_id != first_plan_id:
                    break

            self.assertNotEqual(agent.current_plan.plan_id, first_plan_id)

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
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                planner_config={"planner_interval_steps": 50, "planner_interval_time": 999.0, "planner_trigger_mask": []},
            )
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


    def test_planner_refresh_updates_goal_stack_from_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]

            def fake_decide(_context):
                return BrainDecision(
                    selected_action=ExecutableActionType.WAIT,
                    goal_update="align_with_team_plan",
                    plan_method_id="pm_consult_and_align",
                    next_steps=["consult artifact", "align local schedule"],
                    reason_summary="test planner output",
                    confidence=0.9,
                )

            sim.brain_provider.decide = fake_decide
            sim.update(0.2)

            self.assertEqual(agent.goal, "align_with_team_plan")
            self.assertTrue(any("Planner next steps" in e for e in agent.activity_log))

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



class TestContextGroundingAndAffordances(unittest.TestCase):
    def test_brain_context_contains_grounded_state_and_uncertainty_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            packet = sim.brain_context_builder.build(sim, agent)

            self.assertIn("built_state", packet.world_snapshot)
            self.assertIn("affordance_map", packet.world_snapshot)
            self.assertIn("build_readiness", packet.individual_cognitive_state)
            self.assertIn("certainty_signals", packet.individual_cognitive_state)
            self.assertIn("externalized_artifacts", packet.team_state)
            self.assertIn("semantic_plan_evolution", packet.history_bands)

    def test_legal_affordances_use_accessible_targets_not_impassable_object_centers(self):
        env = Environment(phases=[])
        agent = Agent(name="Engineer", role="Engineer", position=env.get_spawn_point("Engineer"))
        affordances = BrainContextBuilder()._affordances(agent, env)

        interaction_affordances = [
            a for a in affordances
            if a.get("action_type") in {
                ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
                ExecutableActionType.START_CONSTRUCTION.value,
            }
        ]
        self.assertTrue(interaction_affordances)
        for affordance in interaction_affordances:
            self.assertIsNotNone(affordance.get("target_point"))
            self.assertTrue(env.is_point_navigable(tuple(affordance["target_point"])))


class TestLocalBackendFallback(unittest.TestCase):
    def test_local_backend_failure_falls_back_to_rule_brain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="local_http")
            agent = sim.agents[0]
            packet = sim.brain_context_builder.build(sim, agent)

            with patch("modules.brain_provider.request.urlopen", side_effect=error.URLError("offline")):
                decision = sim.brain_provider.decide(packet)

            self.assertIsInstance(decision, BrainDecision)
            self.assertIn(decision.selected_action, list(ExecutableActionType))

    def test_malformed_local_backend_response_is_handled_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="local_http")
            agent = sim.agents[0]
            packet = sim.brain_context_builder.build(sim, agent)

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")

            with patch("modules.brain_provider.request.urlopen", return_value=FakeResponse()):
                decision = sim.brain_provider.decide(packet)

            self.assertIsInstance(decision, BrainDecision)
            self.assertNotEqual(decision.reason_summary, "")


    def test_invalid_local_decision_payload_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="local_http")
            agent = sim.agents[0]
            packet = sim.brain_context_builder.build(sim, agent)

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    payload = {"choices": [{"message": {"content": '{"selected_action":"not_real"}'}}]}
                    return json.dumps(payload).encode("utf-8")

            with patch("modules.brain_provider.request.urlopen", return_value=FakeResponse()):
                decision = sim.brain_provider.decide(packet)

            self.assertIsInstance(decision, BrainDecision)
            self.assertTrue(sim.brain_provider.last_outcome.get("fallback"))

    def test_local_backend_selection_does_not_break_headless_simulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="local_http")
            with patch("modules.brain_provider.request.urlopen", side_effect=error.URLError("offline")):
                sim.update(0.2)
            self.assertGreater(sim.time, 0.0)


if __name__ == "__main__":
    unittest.main()
