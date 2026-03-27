import unittest

from modules.action_schema import ExecutableActionType
from modules.brain_context import BrainContextPacket
from modules.brain_provider import RuleBrain, RuleBrainPolicyConfig, _request_from_context_packet


class TestRuleBrainHierarchicalPolicy(unittest.TestCase):
    def _context(self, affordances=None, control_state=None, known_gaps=None, ready=False, repeated=0, mismatches=None):
        affordances = affordances or [
            {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.6, "target_id": "Team_Info"},
            {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "utility": 0.4, "target_id": "nearby_agent"},
            {"action_type": ExecutableActionType.OBSERVE_ENVIRONMENT.value, "utility": 0.2},
            {"action_type": ExecutableActionType.WAIT.value, "utility": 0.1},
        ]
        return BrainContextPacket(
            static_task_context={"role": "Engineer"},
            world_snapshot={
                "sim_time": 12.0,
                "phase_profile": {"stage": "execution", "name": "default"},
                "built_state": [{"state": "in_progress", "progress": 0.2, "needs_repair": False}],
            },
            individual_cognitive_state={
                "traits": {"communication_propensity": 0.7},
                "known_gaps": list(known_gaps or []),
                "build_readiness": {"ready_for_build": ready},
                "goal_stack": [{"goal_id": "g1"}],
                "loop_counters": {"action_repeats": repeated, "selected_action_repeats": repeated},
                "seconds_since_dik_change": 9.0,
                "control_state": dict(control_state or {"mode": "BOOTSTRAP", "mode_dwell_steps": 0}),
                "inspect_state": {"source_exhaustion": {}},
            },
            team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
            history_bands={"semantic_plan_evolution": {"unresolved_contradictions": list(mismatches or [])}},
            action_affordances=affordances,
        )

    def test_policy_returns_legal_action(self):
        brain = RuleBrain()
        context = self._context()
        decision = brain.decide(context)
        legal = {a["action_type"] for a in context.action_affordances}
        self.assertIn(decision.selected_action.value, legal)

    def test_control_state_persists_and_dwell_increases(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=3))
        context = self._context(control_state={"mode": "ACQUIRE_DIK", "mode_dwell_steps": 0})
        brain.decide(context)
        self.assertEqual(context.individual_cognitive_state["control_state"]["mode"], "ACQUIRE_DIK")
        self.assertEqual(context.individual_cognitive_state["control_state"]["mode_dwell_steps"], 1)
        brain.decide(context)
        self.assertEqual(context.individual_cognitive_state["control_state"]["mode_dwell_steps"], 2)

    def test_hysteresis_reduces_mode_thrash(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=2))
        context = self._context(control_state={"mode": "CONSTRUCT", "mode_dwell_steps": 0}, ready=True,
                                affordances=[
                                    {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.8},
                                    {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.7},
                                ])
        brain.decide(context)
        self.assertEqual(context.individual_cognitive_state["control_state"]["mode"], "CONSTRUCT")

    def test_recovery_mode_activates_under_loop_pressure(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0))
        context = self._context(repeated=6, mismatches=["mismatch"],
                                affordances=[
                                    {"action_type": ExecutableActionType.REASSESS_PLAN.value, "utility": 0.4},
                                    {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.4},
                                ])
        brain.decide(context)
        self.assertEqual(context.individual_cognitive_state["control_state"]["mode"], "RECOVERY")

    def test_weighting_responds_to_context(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0, mode_selection_temperature=0.15))
        context = self._context(ready=True, affordances=[
            {"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "utility": 0.7},
            {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.8},
        ])
        decision = brain.decide(context)
        self.assertIn(decision.selected_action, {ExecutableActionType.TRANSPORT_RESOURCES, ExecutableActionType.START_CONSTRUCTION})

    def test_decide_and_generate_plan_share_policy_core(self):
        brain = RuleBrain(RuleBrainPolicyConfig(mode_selection_temperature=0.2, action_selection_temperature=0.2))
        context = self._context(ready=True, affordances=[
            {"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "utility": 0.8, "target_id": "resource_zone_to_work_zone"},
            {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.7, "target_id": "Build_Table_A"},
        ])
        decision = brain.decide(context)
        request = _request_from_context_packet(context)
        response = brain.generate_plan(request)
        self.assertIn(response.plan.next_action.action_type.value, {ExecutableActionType.TRANSPORT_RESOURCES.value, ExecutableActionType.START_CONSTRUCTION.value})
        self.assertIn(decision.selected_action.value, {ExecutableActionType.TRANSPORT_RESOURCES.value, ExecutableActionType.START_CONSTRUCTION.value})
        self.assertTrue(any("mode=" in note for note in response.plan.notes))
        self.assertTrue(any("policy_snapshot=" in note for note in response.plan.notes))

    def test_request_from_context_carries_control_state_snapshot(self):
        context = self._context(control_state={
            "mode": "REPAIR",
            "previous_mode": "VALIDATE",
            "mode_dwell_steps": 3,
            "last_transition_reason": "contradiction_repair_bias",
            "recovery_active": True,
            "last_policy_snapshot": {"top_features": {"repair_pressure": 0.8}},
        })
        request = _request_from_context_packet(context)
        self.assertEqual(request.control_mode, "REPAIR")
        self.assertEqual(request.previous_control_mode, "VALIDATE")
        self.assertEqual(request.mode_dwell_steps, 3)
        self.assertEqual(request.last_transition_reason, "contradiction_repair_bias")
        self.assertEqual(request.control_state_snapshot.get("mode"), "REPAIR")

    def test_generate_plan_prefers_request_control_state_over_bootstrap_default(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=3, mode_selection_temperature=0.1))
        context = self._context(
            control_state={"mode": "CONSTRUCT", "mode_dwell_steps": 1},
            ready=False,
            affordances=[
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.9, "target_id": "Team_Info"},
                {"action_type": ExecutableActionType.WAIT.value, "utility": 0.1},
            ],
        )
        request = _request_from_context_packet(context)
        request.control_mode = "CONSTRUCT"
        request.mode_dwell_steps = 1
        request.control_state_snapshot = {"mode": "CONSTRUCT", "mode_dwell_steps": 1}
        response = brain.generate_plan(request)
        self.assertTrue(any("'previous_mode': 'CONSTRUCT'" in note for note in response.plan.notes))

    def test_bootstrap_exits_when_shared_source_exhausted_and_role_gap_remaining(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0, mode_selection_temperature=0.1))
        context = self._context(
            control_state={"mode": "BOOTSTRAP", "mode_dwell_steps": 5},
            known_gaps=["missing_role_packet"],
            affordances=[
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.7, "target_id": "Team_Info"},
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.6, "target_id": "Architect_Info"},
            ],
        )
        context.static_task_context["role"] = "Architect"
        context.individual_cognitive_state["inspect_state"] = {
            "source_exhaustion": {
                "Team_Info": {"exhausted": True, "no_new_dik_streak": 3, "inspected": True},
                "Architect_Info": {"exhausted": False, "inspected": False},
            }
        }
        brain.decide(context)
        self.assertNotEqual(context.individual_cognitive_state["control_state"]["mode"], "BOOTSTRAP")

    def test_control_state_transition_history_and_snapshot(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0))
        context = self._context(
            control_state={"mode": "BOOTSTRAP", "mode_dwell_steps": 4, "mode_history": [], "transition_history": []},
            ready=True,
            affordances=[
                {"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "utility": 0.8, "target_id": "resource_zone_to_work_zone"},
                {"action_type": ExecutableActionType.START_CONSTRUCTION.value, "utility": 0.7, "target_id": "Build_Table_A"},
            ],
        )
        brain.decide(context)
        control = context.individual_cognitive_state["control_state"]
        self.assertEqual(control["previous_mode"], "BOOTSTRAP")
        self.assertIn(control["mode"], {"LOGISTICS", "CONSTRUCT"})
        self.assertEqual(control["mode_dwell_steps"], 1)
        self.assertTrue(control["last_transition_reason"])
        self.assertTrue(control["mode_history"])
        self.assertTrue(control["transition_history"])
        self.assertIn("selected_action", control.get("last_policy_snapshot", {}))

    def test_non_transition_tick_increments_dwell(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=4))
        context = self._context(
            control_state={"mode": "ACQUIRE_DIK", "mode_dwell_steps": 1, "mode_history": [], "transition_history": []},
            known_gaps=["g1", "g2", "g3"],
            affordances=[
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.8, "target_id": "Team_Info"},
                {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "utility": 0.7, "target_id": "Architect"},
            ],
        )
        brain.decide(context)
        control = context.individual_cognitive_state["control_state"]
        self.assertEqual(control["mode"], "ACQUIRE_DIK")
        self.assertEqual(control["mode_dwell_steps"], 2)
        self.assertFalse(control.get("transition_history"))

    def test_team_info_exhaustion_switches_to_role_specific_method_and_cooldown(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0))
        context = self._context(
            control_state={"mode": "ACQUIRE_DIK", "mode_dwell_steps": 2},
            known_gaps=["missing_role_packet"],
            affordances=[
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.9, "target_id": "Team_Info"},
                {"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.5, "target_id": "Engineer_Info"},
            ],
        )
        context.individual_cognitive_state["inspect_state"] = {
            "source_exhaustion": {
                "Team_Info": {"exhausted": True, "no_new_dik_streak": 3, "inspected": True},
                "Engineer_Info": {"exhausted": False, "inspected": False},
            }
        }
        brain.decide(context)
        method_state = context.individual_cognitive_state["control_state"].get("method_state", {})
        self.assertEqual(method_state.get("active_method_id"), "AcquireRoleSpecificGrounding")
        self.assertIn("Team_Info", method_state.get("source_cooldowns", {}))

    def test_active_method_persists_across_ticks(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0))
        context = self._context(
            control_state={"mode": "LOGISTICS", "mode_dwell_steps": 2},
            ready=True,
            affordances=[{"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "utility": 0.8, "target_id": "resource_zone_to_work_zone"}],
        )
        brain.decide(context)
        first_method = context.individual_cognitive_state["control_state"].get("method_state", {}).get("active_method_id")
        brain.decide(context)
        second_state = context.individual_cognitive_state["control_state"].get("method_state", {})
        self.assertEqual(first_method, second_state.get("active_method_id"))
        self.assertGreaterEqual(int(second_state.get("step_started_tick", 0)), 0)

    def test_mode_method_consistency(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0))
        context = self._context(
            control_state={"mode": "VALIDATE", "mode_dwell_steps": 3},
            affordances=[
                {"action_type": ExecutableActionType.VALIDATE_CONSTRUCTION.value, "utility": 0.8},
                {"action_type": ExecutableActionType.OBSERVE_ENVIRONMENT.value, "utility": 0.2},
            ],
        )
        brain.decide(context)
        control = context.individual_cognitive_state["control_state"]
        method_id = control.get("method_state", {}).get("active_method_id")
        self.assertIn(method_id, {"ValidateProject", None})

    def test_generate_plan_carries_method_notes(self):
        brain = RuleBrain(RuleBrainPolicyConfig(mode_selection_temperature=0.1, action_selection_temperature=0.1))
        context = self._context(
            control_state={"mode": "ACQUIRE_DIK", "mode_dwell_steps": 2},
            affordances=[{"action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, "utility": 0.9, "target_id": "Engineer_Info"}],
        )
        response = brain.generate_plan(_request_from_context_packet(context))
        self.assertTrue(any("method=" in note for note in response.plan.notes))


if __name__ == "__main__":
    unittest.main()
