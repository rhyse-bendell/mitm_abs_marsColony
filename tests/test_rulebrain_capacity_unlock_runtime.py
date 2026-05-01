import unittest
from modules.action_schema import ExecutableActionType
from modules.brain_context import BrainContextBuilder, BrainContextPacket
from modules.brain_provider import RuleBrain, RuleBrainPolicyConfig
from modules.simulation import Environment
from modules.agent import Agent


class TestRuleBrainCapacityUnlockRuntime(unittest.TestCase):
    def test_context_exposes_bridge_unlock(self):
        env = Environment()
        a = Agent("A", "builder", (0, 0))
        env.construction.get_next_capacity_unlock = lambda: {"buildable_capacity_exhausted": True, "bridge_id": "bridge_bc", "locked_site_id": "site_c", "locked_remaining_capacity": 16, "bridge_status": "not_started", "required_resources": 20, "delivered_resources": 0}
        sim = type("S", (), {"environment": env, "time": 0.0, "agents": [a], "team_knowledge_manager": type("TK", (), {"artifacts": {}, "summarize": lambda self: {}, "recent_updates": []})(), "task_model": None})()
        packet = BrainContextBuilder().build(sim, a)
        self.assertEqual(packet.world_snapshot["capacity_unlock"]["bridge_id"], "bridge_bc")
        self.assertTrue(any(x.get("action_type") == "transport_resources" and x.get("target_id") == "bridge_bc" for x in packet.action_affordances))

    def test_rulebrain_bridge_selection_transport_and_build(self):
        brain = RuleBrain(RuleBrainPolicyConfig(min_mode_dwell_steps=0, mode_selection_temperature=0.01, action_selection_temperature=0.01))
        base = {
            "sim_time": 10.0,
            "phase_profile": {"stage": "execution"},
            "built_state": [],
            "capacity_unlock": {"buildable_capacity_exhausted": True, "bridge_id": "bridge_bc", "locked_site_id": "site_c", "locked_remaining_capacity": 16, "bridge_status": "not_started", "required_resources": 20, "delivered_resources": 0},
        }
        ctx = BrainContextPacket(
            static_task_context={"role": "Engineer"}, world_snapshot=base,
            individual_cognitive_state={"build_readiness": {"ready_for_build": True}, "known_gaps": [], "loop_counters": {}, "progress_state": {}, "goal_stack": [{"goal_id": "phase1"}], "inspect_state": {"source_exhaustion": {}}, "epistemic_sufficiency": {}},
            team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "team_shared_knowledge": {}}, history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
            action_affordances=[{"action_type": "reassess_plan", "utility": 0.9}, {"action_type": "transport_resources", "target_id": "bridge_bc", "reachable": True, "utility": 0.6}],
        )
        d1 = brain.decide(ctx)
        self.assertEqual(d1.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)
        self.assertEqual(d1.target_id, "bridge_bc")

        base2 = dict(base)
        base2["capacity_unlock"] = {**base["capacity_unlock"], "bridge_status": "in_progress", "delivered_resources": 20}
        ctx2 = BrainContextPacket(
            static_task_context={"role": "Engineer"}, world_snapshot=base2,
            individual_cognitive_state=ctx.individual_cognitive_state, team_state=ctx.team_state, history_bands=ctx.history_bands,
            action_affordances=[{"action_type": "reassess_plan", "utility": 0.9}, {"action_type": "continue_construction", "target_id": "bridge_bc", "reachable": True, "utility": 0.5}],
        )
        d2 = brain.decide(ctx2)
        self.assertIn(d2.selected_action, {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION})
        self.assertEqual(d2.target_id, "bridge_bc")

    def test_bridge_completion_unlocks_site_c(self):
        env = Environment()
        b = env.construction.bridges["bridge_bc"]
        env.construction.build_bridge_bc(quantity=b.required_resources)
        self.assertEqual(env.construction.bridges["bridge_bc"].status, "complete")
        self.assertTrue(env.construction._is_site_buildable("site_c"))
        pid, reason = env.construction.create_project("site_c", "house")
        self.assertIsNotNone(pid)
        self.assertEqual(reason, "created")


if __name__ == "__main__":
    unittest.main()
