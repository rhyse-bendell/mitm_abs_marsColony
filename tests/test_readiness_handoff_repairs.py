import unittest
from unittest import mock

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class ReadinessHandoffRepairTests(unittest.TestCase):
    def _prime_min_readiness(self, sim, agent):
        team_packet = sim.environment.knowledge_packets["Team_Info"]
        role_source = f"{agent.role}_Info"
        role_packet = sim.environment.knowledge_packets.get(role_source, {})
        for info in list(team_packet.get("information", []))[:2]:
            agent.mental_model["information"].add(info)
        for info in list(role_packet.get("information", []))[:1]:
            agent.mental_model["information"].add(info)
        if "R_HOUSE_VALIDITY" not in agent.mental_model["knowledge"].rules:
            agent.mental_model["knowledge"].rules.append("R_HOUSE_VALIDITY")
        for source_id in ["Team_Info", role_source, "Engineer_Info"]:
            agent.source_inspection_state[source_id] = "inspected"
            mem = agent.source_memory_state.setdefault(source_id, {})
            mem["ever_inspected"] = True
            mem["last_inspected_time"] = float(sim.time)
            mem["last_verified_time"] = float(sim.time)
            mem["memory_confidence"] = 0.9

    def test_inspect_dik_refreshes_project_local_grounding_marker(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_min_readiness(sim, agent)
        role_source = f"{agent.role}_Info"
        role_mem = agent.source_memory_state.setdefault(role_source, {})
        role_mem["last_verified_time"] = float(sim.time) - 300.0
        role_mem["memory_confidence"] = 0.2
        before = set(agent._build_readiness_blockers(sim.environment, sim_state=sim))
        self.assertIn("stale_epistemic_grounding", before)

        target = sim.environment.get_interaction_target_position(role_source, from_position=agent.position)
        agent.position = target
        ok = agent._inspect_source(sim.environment, role_source, sim_state=sim)
        self.assertTrue(ok)
        self.assertTrue(any(e.get("event_type") == "epistemic_grounding_refreshed_after_inspect" for e in sim.logger.recent_events))
        sim.stop()

    def test_readiness_unlocks_and_followup_prefers_executable_build_or_validate(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_min_readiness(sim, agent)
        project_id = "Build_Table_B"
        project = sim.environment.construction.projects[project_id]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)

        agent.post_inspect_handoff = {"pending": True, "dik_changed": True, "readiness_changed": True, "source_id": f"{agent.role}_Info", "expires_at": sim.time + 5.0}
        agent._recompute_project_state_after_dik_change(sim_state=sim, trigger_source="unit")
        blockers = set(agent._build_readiness_blockers(sim.environment, sim_state=sim))
        self.assertNotIn("stale_epistemic_grounding", blockers)
        with mock.patch.object(agent, "_critical_unmet_source_targets", return_value={}), mock.patch.object(agent, "_cross_role_engineer_dependency_gap", return_value=False), mock.patch.object(agent, "_team_plan_requires_uptake", return_value=False):
            decision = agent._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertIn(decision.selected_action, {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION, ExecutableActionType.VALIDATE_CONSTRUCTION})

        for _ in range(len(project.get("build_steps") or [])):
            sim.environment.construction.execute_build_step(project_id, actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="claim", note="c", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="evidence", note="e", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="design_note", note="d", actor=agent.name, sim_time=sim.time)
        with mock.patch.object(agent, "_critical_unmet_source_targets", return_value={}), mock.patch.object(agent, "_cross_role_engineer_dependency_gap", return_value=False), mock.patch.object(agent, "_team_plan_requires_uptake", return_value=False):
            decision2 = agent._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertEqual(decision2.selected_action, ExecutableActionType.VALIDATE_CONSTRUCTION)
        sim.stop()

    def test_validation_accepts_semantic_dependency_support_when_exact_missing(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["expected_rules"] = ["R_GREENHOUSE_SUPPORT_DEPENDENCY"]
        agent.mental_model["knowledge"].rules.append("R_GREENHOUSE_CONNECTOR_SUPPORT_DEPENDENCY")
        matched, missing = agent._construction_rule_match(project_id, environment=sim.environment, sim_state=sim, include_team=True)
        self.assertTrue(matched)
        self.assertEqual(missing, [])
        sim.stop()

    def test_closure_repair_avoids_exact_unsat_when_semantic_support_exists(self):
        sim = SimulationState(phases=[])
        requester, responder = sim.agents[0], sim.agents[1]
        responder.mental_model["knowledge"].rules.append("R_GREENHOUSE_CONNECTOR_SUPPORT_DEPENDENCY")
        responder.receive_message(
            {
                "type": "TKRQ",
                "sender": requester.name,
                "content": ["rule:R_GREENHOUSE_SUPPORT_DEPENDENCY"],
                "project_id": "Build_Table_A",
            },
            from_agent=requester.name,
            sim_state=sim,
        )
        self.assertFalse(any(e.get("event_type") == "closure_repair_exact_rule_unsatisfiable" for e in sim.logger.recent_events))
        queued = responder.team_plan_state.get("pending_outbound_messages", [])
        self.assertTrue(any(m.get("type") == "TPS" and (m.get("content", {}) or {}).get("semantic_support_available") for m in queued))
        sim.stop()

    def test_transport_pruned_for_materially_satisfied_project_without_new_binding(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        decision = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id=project_id, confidence=0.8)
        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        self.assertNotEqual(translated.get("project_id"), project_id)
        self.assertTrue(any(e.get("event_type") in {"project_transport_target_suppressed", "transport_pruned_materially_satisfied"} for e in sim.logger.recent_events))
        sim.stop()


if __name__ == "__main__":
    unittest.main()
