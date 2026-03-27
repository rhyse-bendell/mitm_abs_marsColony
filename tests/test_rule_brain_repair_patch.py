import unittest
from types import SimpleNamespace

from modules.agent import Agent
from modules.brain_context import BrainContextPacket
from modules.brain_provider import RuleBrain
from modules.simulation import SimulationState


class _CtxBuilder:
    def build(self, _sim, _agent):
        return BrainContextPacket(
            static_task_context={"role": "Engineer"},
            world_snapshot={"sim_time": 1.0, "phase_profile": {"name": "execution"}, "built_state": []},
            individual_cognitive_state={
                "traits": {"communication_propensity": 1.0, "help_tendency": 1.0},
                "known_gaps": ["water_gap"],
                "build_readiness": {"ready_for_build": False},
                "goal_stack": [{"goal_id": "prepare_water"}],
                "loop_counters": {"action_repeats": 0, "selected_action_repeats": 0},
                "seconds_since_dik_change": 10.0,
                "control_state": {"mode": "BOOTSTRAP", "mode_dwell_steps": 1},
                "inspect_state": {"source_exhaustion": {"Team_Info": {"exhausted": True, "no_new_dik_streak": 3}}},
            },
            team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
            history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
            action_affordances=[
                {"action_type": "observe_environment", "utility": 0.6},
                {"action_type": "wait", "utility": 0.2},
            ],
        )


class _Logger:
    def log_event(self, *_args, **_kwargs):
        return None


class TestRuleBrainRepairPatch(unittest.TestCase):
    def test_build_brain_request_includes_compact_control_state_fields(self):
        agent = Agent("Engineer", "Engineer")
        agent.control_state.update(
            {
                "mode": "LOGISTICS",
                "previous_mode": "COORDINATE",
                "mode_dwell_steps": 4,
                "last_transition_reason": "build_ready_incomplete_projects_bias_logistics",
                "last_policy_snapshot": {"top_features": {"build_opportunity": 1.0}},
            }
        )
        sim = SimpleNamespace(
            time=1.0,
            bootstrap_reuse_enabled=False,
            get_agent_brain_runtime=lambda _a: {},
            task_model=SimpleNamespace(task_id="mars_colony"),
        )
        context = _CtxBuilder().build(sim, agent)
        request = agent._build_brain_request(sim, context, request_explanation=False, trigger_reason="unit_test")
        self.assertEqual(request.control_mode, "LOGISTICS")
        self.assertEqual(request.previous_control_mode, "COORDINATE")
        self.assertEqual(request.mode_dwell_steps, 4)
        self.assertEqual(request.last_transition_reason, "build_ready_incomplete_projects_bias_logistics")
        self.assertEqual(request.control_state_snapshot.get("mode"), "LOGISTICS")

    def test_role_source_preferred_after_team_exhaustion(self):
        agent = Agent("Engineer", "Engineer")
        agent.goal_stack = [{"goal": "secure water connectivity", "status": "active"}]
        env = SimpleNamespace(
            knowledge_packets={"Team_Info": {}, "Engineer_Info": {}, "Botanist_Info": {}},
            get_interaction_target_position=lambda source_id, from_position=None: (1.0, 1.0),
        )
        agent.source_inspection_state = {"Team_Info": "inspected", "Engineer_Info": "unseen", "Botanist_Info": "unseen"}
        agent.source_exhaustion_state = {
            "Team_Info": {"exhausted": True, "no_new_dik_streak": 3, "inspect_count": 2},
            "Engineer_Info": {"exhausted": False, "no_new_dik_streak": 0, "inspect_count": 0},
            "Botanist_Info": {"exhausted": False, "no_new_dik_streak": 0, "inspect_count": 0},
        }
        ranked = agent._candidate_information_sources(env)
        self.assertEqual(ranked[0][1], "Engineer_Info")

    def test_local_policy_refresh_for_rule_brain_when_planner_not_due(self):
        agent = Agent("Engineer", "Engineer")
        sim = SimpleNamespace(
            configured_brain_backend="rule_brain",
            brain_provider=RuleBrain(),
            brain_context_builder=_CtxBuilder(),
            get_agent_brain_runtime=lambda _a: {"provider": RuleBrain(), "configured_backend": "rule_brain"},
            logger=_Logger(),
            time=1.0,
        )
        env = SimpleNamespace()
        refreshed = agent._attempt_local_rule_brain_refresh(sim, env, "split_mode_cadence_not_due")
        self.assertTrue(refreshed)
        self.assertTrue(agent.current_action)

    def test_default_agent_traits_are_ideal_baseline(self):
        sim = SimulationState(speed=1.0)
        try:
            self.assertTrue(sim.agents)
            for agent in sim.agents:
                self.assertEqual(agent.communication_propensity, 1.0)
                self.assertEqual(agent.goal_alignment, 1.0)
                self.assertEqual(agent.help_tendency, 1.0)
                self.assertEqual(agent.build_speed, 1.0)
                self.assertEqual(agent.rule_accuracy, 1.0)
        finally:
            if hasattr(sim, "planner_executor"):
                sim.planner_executor.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
