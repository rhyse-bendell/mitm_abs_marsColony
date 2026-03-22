import unittest
from unittest.mock import patch
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
        self.assertTrue(any(g.goal_level == "mission" and g.trust_tier == "canonical" for g in agent.goal_registry.values()))
        sim.stop()

    def test_phase_transition_updates_goal_statuses(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        phase2_goal = agent.goal_registry.get("G_PHASE2_SUPPORT_40_CIV_20_VIP")
        self.assertIsNotNone(phase2_goal)
        agent._update_goal_states_from_runtime(sim, sim.environment)
        self.assertIn(phase2_goal.status, {"inactive", "candidate", "blocked", "active"})
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
        with patch("modules.agent.random.random", return_value=0.0):
            agent._apply_task_derivations(sim_state=None)
        agent._update_goal_states_from_runtime(sim_state=None, environment=env)

        self.assertTrue(any("integrate_new_derivation" in g.label for g in agent.goal_registry.values()))

    def test_mismatch_and_artifact_events_activate_repair_validation_goals(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        agent.activity_log.append("Mismatch with construction: connector rule violated")
        sim.team_knowledge_manager.recent_updates.append({"artifact": "x"})
        agent._update_goal_states_from_runtime(sim, sim.environment)
        labels = {g.label for g in agent.goal_registry.values()}
        self.assertIn("repair_detected_mismatch", labels)
        self.assertIn("validate_externalization", labels)
        self.assertNotIn("consult_artifact", labels)
        sim.stop()

    def test_consult_artifact_requires_baseline_epistemic_sources_completed(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        sim.team_knowledge_manager.externalize_artifact(
            artifact_id="whiteboard:test",
            artifact_type="whiteboard_plan",
            summary="s",
            content={"x": 1},
            author="teammate",
            sim_time=sim.time,
        )
        agent.source_inspection_state["Team_Info"] = "inspected"
        agent.source_inspection_state[f"{agent.role}_Info"] = "in_progress"
        agent._update_goal_states_from_runtime(sim, sim.environment)
        labels = {g.label for g in agent.goal_registry.values() if g.status in {"active", "queued", "candidate"}}
        self.assertNotIn("consult_artifact", labels)

        agent.source_inspection_state[f"{agent.role}_Info"] = "inspected"
        agent._update_goal_states_from_runtime(sim, sim.environment)
        labels = {g.label for g in agent.goal_registry.values() if g.status in {"active", "queued", "candidate"}}
        self.assertIn("consult_artifact", labels)
        sim.stop()

    def test_support_goal_activation_is_deduplicated(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        sim.team_knowledge_manager.externalize_artifact(
            artifact_id="whiteboard:test2",
            artifact_type="whiteboard_plan",
            summary="s2",
            content={"y": 2},
            author="teammate",
            sim_time=sim.time,
        )
        agent.source_inspection_state["Team_Info"] = "inspected"
        agent.source_inspection_state[f"{agent.role}_Info"] = "inspected"
        agent._update_goal_states_from_runtime(sim, sim.environment)
        agent._update_goal_states_from_runtime(sim, sim.environment)
        support_consult = [g for g in agent.goal_registry.values() if g.goal_id == "SUPPORT_CONSULT_ARTIFACT"]
        self.assertEqual(len(support_consult), 1)
        sim.stop()

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

    def test_plan_method_id_validated_against_task_methods(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        context = BrainContextBuilder().build(sim, agent)
        req = AgentBrainRequest(
            request_id="r2",
            tick=2,
            sim_time=0.2,
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
                "response_id": "y",
                "agent_id": agent.agent_id,
                "plan": {
                    "plan_id": "p2",
                    "plan_horizon": 2,
                    "ordered_goals": [{"goal_id": "G_PLANNING_REDUCE_UNCERTAINTY", "description": "Reduce uncertainty and plan", "priority": 0.6, "status": "active"}],
                    "ordered_actions": [{"step_index": 0, "action_type": "inspect_information_source", "expected_purpose": "inspect"}],
                    "next_action": {"step_index": 0, "action_type": "inspect_information_source", "expected_purpose": "inspect"},
                    "plan_method_id": "PM_REVIEW_SHARED_INFO",
                    "confidence": 0.7,
                },
            }
        )
        grounded = agent._validate_and_ground_response_plan(response, context, req)
        self.assertIn(getattr(grounded.plan, "_method_status", ""), {"accepted", "low_trust"})
        sim.stop()

    def test_plan_next_action_can_be_rejected_without_breaking_sim(self):
        sim = SimulationState(phases=[], brain_backend="local_stub")
        for _ in range(2):
            sim.update(0.1)
        self.assertIsNotNone(sim.agents[0].current_plan)
        sim.stop()


if __name__ == "__main__":
    unittest.main()
