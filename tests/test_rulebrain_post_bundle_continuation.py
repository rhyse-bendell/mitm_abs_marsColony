import unittest

from modules.action_schema import ExecutableActionType
from modules.brain_context import BrainContextPacket
from modules.brain_provider import RuleBrain, RuleBrainPolicyConfig


class TestRuleBrainPostBundleContinuation(unittest.TestCase):
    def _context(self, built_state, affordances):
        return BrainContextPacket(
            static_task_context={"role": "Engineer"},
            world_snapshot={"sim_time": 120.0, "phase_profile": {"stage": "execution"}, "built_state": built_state},
            individual_cognitive_state={
                "build_readiness": {"ready_for_build": True},
                "known_gaps": [],
                "loop_counters": {"no_progress_streak": 2},
                "progress_state": {"no_progress_streak": 2},
                "goal_stack": [{"goal_id": "phase1"}],
                "control_state": {"mode": "LOGISTICS", "mode_dwell_steps": 1},
                "inspect_state": {"source_exhaustion": {}},
            },
            team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "team_shared_knowledge": {}},
            history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
            action_affordances=affordances,
        )

    def test_prefers_transport_for_materially_incomplete_unbuilt_project(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0, mode_selection_temperature=0.01, action_selection_temperature=0.01))
        built_state = [
            {"structure_id": "site_b_house_002", "project_status": "in_progress", "required_resources": 10, "delivered_resources": 6, "resource_complete": False, "physical_build_progress": 0.0, "state": "in_progress", "progress": 0.0},
        ]
        ctx = self._context(
            built_state,
            affordances=[
                {"action_type": ExecutableActionType.REASSESS_PLAN.value, "utility": 0.9},
                {"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "utility": 0.5, "reachable": True},
                {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.4, "reachable": True},
            ],
        )
        decision = brain.decide(ctx)
        self.assertEqual(decision.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)

    def test_prefers_start_or_continue_when_materially_ready_unbuilt(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0, mode_selection_temperature=0.01, action_selection_temperature=0.01))
        built_state = [
            {"structure_id": "site_b_house_002", "project_status": "in_progress", "required_resources": 10, "delivered_resources": 10, "resource_complete": True, "physical_build_progress": 0.0, "state": "in_progress", "progress": 0.0},
        ]
        ctx = self._context(
            built_state,
            affordances=[
                {"action_type": ExecutableActionType.REASSESS_PLAN.value, "utility": 0.95},
                {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.45, "reachable": True},
                {"action_type": ExecutableActionType.CONTINUE_CONSTRUCTION.value, "utility": 0.4, "reachable": True},
            ],
        )
        decision = brain.decide(ctx)
        self.assertIn(decision.selected_action, {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION})

    def test_reassess_remains_when_no_executable_transport_or_build(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0, mode_selection_temperature=0.01, action_selection_temperature=0.01))
        built_state = [
            {"structure_id": "site_b_house_002", "project_status": "in_progress", "required_resources": 10, "delivered_resources": 6, "resource_complete": False, "physical_build_progress": 0.0, "state": "in_progress", "progress": 0.0},
        ]
        ctx = self._context(
            built_state,
            affordances=[
                {"action_type": ExecutableActionType.REASSESS_PLAN.value, "utility": 0.8},
                {"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "utility": 0.9, "reachable": False},
                {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.9, "reachable": False},
            ],
        )
        decision = brain.decide(ctx)
        self.assertEqual(decision.selected_action, ExecutableActionType.REASSESS_PLAN)

