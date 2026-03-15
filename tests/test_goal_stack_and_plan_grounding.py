import unittest
from modules.agent import Agent
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse
from modules.brain_context import BrainContextBuilder
from modules.environment import Environment
from modules.simulation import SimulationState
from modules.task_model import load_task_model


class GoalStackAndPlanGroundingTests(unittest.TestCase):
    def test_task_goals_seed_into_registry(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        task_goal_ids = {g.goal_id for g in sim.task_model.goals.values() if g.enabled}
        self.assertTrue(task_goal_ids.intersection(set(agent.goal_registry.keys())))
        self.assertTrue(any(g.status in {"active", "queued", "candidate"} for g in agent.goal_registry.values()))
        sim.stop()

    def test_dik_derivation_can_satisfy_or_unlock_goal(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        agent = Agent(name="Engineer", role="Engineer", position=(8.0, 6.6))
        agent.task_model = model
        agent._seed_task_defined_goals()

        team_packet = env.knowledge_packets["Team_Info"]
        wanted = {"D_SHARED_PLANNING_REVIEW_ORDER", "D_SHARED_PHASE1_TARGET_50_CIV", "D_SHARED_PHASE2_TARGET_40_CIV_20_VIP"}
        for d in team_packet["data"]:
            if d.id in wanted:
                agent.mental_model["data"].add(d)
        agent._apply_task_derivations(sim_state=None)
        agent._update_goal_states_from_runtime(sim_state=None, environment=env)

        self.assertTrue(any("integrate_new_derivation" in g.label for g in agent.goal_registry.values()))

    def test_unknown_plan_method_is_mapped_safely(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        context = BrainContextBuilder().build(sim, agent)
        req = AgentBrainRequest(
            request_id="r1",
            tick=1,
            sim_time=0.1,
            agent_id=agent.agent_id,
            display_name=agent.display_name,
            task_id="mars_colony",
            phase="Phase 1",
            local_context_summary="ctx",
            local_observations=[],
            working_memory_summary={},
            inbox_summary=[],
            current_goal_stack=[],
            current_plan_summary={},
            allowed_actions=context.action_affordances,
            planning_horizon_config={"max_steps": 2},
            request_explanation=False,
        )
        response = AgentBrainResponse.from_dict(
            {
                "response_id": "x",
                "agent_id": agent.agent_id,
                "plan": {
                    "plan_id": "p1",
                    "plan_horizon": 2,
                    "ordered_goals": [{"goal_id": "G_ACQUIRE_MISSING_DIK", "description": "Acquire missing DIK", "priority": 0.7, "status": "active"}],
                    "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "hold"}],
                    "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "hold"},
                    "plan_method_id": "PM_DOES_NOT_EXIST",
                    "confidence": 0.8,
                },
            }
        )
        grounded = agent._validate_and_ground_response_plan(response, context, req)
        self.assertNotEqual(grounded.plan.plan_method_id, "PM_DOES_NOT_EXIST")
        self.assertIn("unknown_method_fallback", grounded.plan.notes)
        sim.stop()

    def test_plan_next_action_can_be_rejected_without_breaking_sim(self):
        sim = SimulationState(phases=[], brain_backend="local_stub")
        for _ in range(2):
            sim.update(0.1)
        self.assertIsNotNone(sim.agents[0].current_plan)
        sim.stop()


if __name__ == "__main__":
    unittest.main()
