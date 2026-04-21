import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.agent import Agent
from modules.simulation import SimulationState
from modules.task_model import normalize_rule_token


class _MemoryLogger:
    def __init__(self):
        self.events = []

    def log_event(self, time, event_type, payload):
        self.events.append((time, event_type, payload))


class TestValidationDialogueProtocol(unittest.TestCase):
    def test_blockage_opens_validation_question_record(self):
        sim = SimulationState(phases=[])
        try:
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project["started"] = True
            project["status"] = "ready_for_validation"
            updates, _ = sim._sync_validation_discussions(trigger_event="unit_test")
            self.assertGreaterEqual(updates, 1)
            discussion = project.get("validation_discussion")
            self.assertIsInstance(discussion, dict)
            self.assertIn(discussion.get("status"), {"open", "active_discussion", "provisionally_supported"})
            self.assertTrue(discussion.get("candidate_claim"))
        finally:
            sim.stop()

    def test_validation_request_response_is_local_dik_bounded(self):
        sim = SimulationState(phases=[])
        try:
            a = sim.agents[0]
            b = sim.agents[1]
            project_id = "Build_Table_A"
            rid = normalize_rule_token(sim.environment.construction.projects[project_id]["expected_rules"][0])
            b.mental_model["knowledge"].add_rule(rid, [])
            msg = {
                "type": "TKRQ",
                "sender": a.name,
                "project_id": project_id,
                "closure_blocker_signature": f"{project_id}|missing|{rid}",
                "content": [f"rule:{rid}"],
                "request_modes": ["exact_rule"],
            }
            b.receive_message(msg, from_agent=a.name, sim_state=sim)
            pending = [m for m in b.team_plan_state.get("pending_outbound_messages", []) if m.get("type") == "TPS"]
            self.assertTrue(pending)
            self.assertEqual(pending[-1]["content"].get("response_category"), "state_support")
        finally:
            sim.stop()

    def test_externalized_support_updates_discussion(self):
        sim = SimulationState(phases=[])
        try:
            project_id = "Build_Table_A"
            cm = sim.environment.construction
            cm.record_validation_dialogue_event(project_id, event_type="validation_support_externalized", actor="Engineer", payload={"evidence": "rule"}, sim_time=sim.time)
            discussion = cm.projects[project_id].get("validation_discussion")
            self.assertGreaterEqual(len(discussion.get("support_items", [])), 1)
        finally:
            sim.stop()

    def test_externalized_support_unlocks_validation_blocker(self):
        sim = SimulationState(phases=[])
        try:
            agent = sim.agents[0]
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project["started"] = True
            project["status"] = "ready_for_validation"
            rid = normalize_rule_token(project["expected_rules"][0])
            agent.mental_model["knowledge"].add_rule(rid, [])
            decision = SimpleNamespace(selected_action=SimpleNamespace(value="validate_construction"))
            from modules.action_schema import BrainDecision, ExecutableActionType
            decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id)
            blockers_before, _ = agent._construction_action_blockers(decision, {"project_id": project_id}, sim.environment, sim_state=sim)
            self.assertIn("insufficient_externalized_validation_support", blockers_before)
            sim.environment.construction.record_validation_dialogue_event(project_id, event_type="validation_support_externalized", actor=agent.name, payload={"rule": rid}, sim_time=sim.time)
            blockers_after, _ = agent._construction_action_blockers(decision, {"project_id": project_id}, sim.environment, sim_state=sim)
            self.assertNotIn("insufficient_externalized_validation_support", blockers_after)
        finally:
            sim.stop()

    def test_non_owner_can_still_run_mismatch_detection_during_discussion(self):
        sim = SimulationState(phases=[])
        try:
            captured = []
            sim.logger.register_event_listener(lambda e: captured.append(e))
            agent = sim.agents[0]
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project.update(
                {
                    "started": True,
                    "in_progress": True,
                    "status": "ready_for_validation",
                    "closure_owner": "OtherAgent",
                    "closure_status": "assigned",
                    "required_resources": {"bricks": 4},
                    "delivered_resources": {"bricks": 4},
                    "correct": True,
                }
            )
            sim.environment.construction.ensure_validation_discussion(project_id, sim_time=sim.time)
            with patch("random.random", return_value=0.0):
                agent.compare_and_repair_construction(sim.environment.construction, sim_state=sim)
            self.assertTrue(any(str(evt.get("event_type")) == "mismatch_detection_allowed_during_validation_discussion" for evt in captured))
        finally:
            sim.stop()

    def test_stagnation_recovery_queues_uncertainty_message(self):
        sim = SimulationState(phases=[])
        try:
            agent = sim.agents[0]
            project_id = "Build_Table_A"
            project = sim.environment.construction.projects[project_id]
            project["started"] = True
            project["status"] = "ready_for_validation"
            agent.project_closure_state.update({"active": True, "project_id": project_id, "commit_until": sim.time + 30.0})
            goal = agent._activate_support_goal("refresh_assumptions", "unit_test", sim_state=sim)
            recurrence_key = f"{goal.goal_id}:assumptions_currently_fresh"
            agent.support_goal_nonexec_counts[recurrence_key] = 2
            original_support_exec = agent._support_goal_executable
            with patch.object(agent, "_support_goal_executable", wraps=agent._support_goal_executable) as wrapped:
                def _forced(goal_obj, _sim, _env):
                    if goal_obj.goal_id == goal.goal_id:
                        return False, "assumptions_currently_fresh"
                    return original_support_exec(goal_obj, _sim, _env)
                wrapped.side_effect = _forced
                agent._update_goal_states_from_runtime(sim, sim.environment)
            pending = [m for m in agent.team_plan_state.get("pending_outbound_messages", []) if m.get("type") == "TPS"]
            self.assertTrue(any((m.get("content") or {}).get("state_event") == "validation_uncertainty" for m in pending))
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
