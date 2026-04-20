import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class TestDIKCommunicationFlow(unittest.TestCase):
    def _prime_readiness_without_rules(self, sim, agent):
        packet = sim.environment.knowledge_packets["Team_Info"]
        agent.mental_model["information"].add(packet["information"][0])
        agent.mental_model["information"].add(packet["information"][1])
        agent.source_inspection_state["Team_Info"] = "inspected"

    def test_communication_only_shares_requested_dik(self):
        sim = SimulationState(phases=[])
        sender, receiver = sim.agents[0], sim.agents[1]
        sender.position = receiver.position = (8.0, 6.6)

        sender.mental_model["knowledge"].add_rule("R_REQ", [])
        sender.mental_model["knowledge"].add_rule("R_EXTRA", [])

        sender.receive_message(
            {"type": "TKRQ", "sender": receiver.name, "content": ["rule:R_REQ"]},
            from_agent=receiver.name,
            sim_state=sim,
        )
        sender.communicate_with(receiver, sim_state=sim)

        self.assertIn("R_REQ", receiver.mental_model["knowledge"].rules)
        self.assertNotIn("R_EXTRA", receiver.mental_model["knowledge"].rules)
        sim.stop()

    def test_blocked_validation_emits_specific_request_and_response_unblocks(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)

        project_id = "Build_Table_B"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_REQ"]

        self._prime_readiness_without_rules(sim, owner)
        helper.mental_model["knowledge"].add_rule("R_REQ", [])

        decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
        translated = owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        self.assertEqual(translated[0]["type"], "communicate")
        self.assertIn("missing_expected_rule:R_REQ", owner.known_gaps)

        request_messages = owner.generate_message(recipient_name=helper.name)
        self.assertTrue(any(m["type"] == "TKRQ" and "rule:R_REQ" in m["content"] for m in request_messages))

        helper.receive_message({"type": "TKRQ", "sender": owner.name, "content": ["rule:R_REQ"]}, from_agent=owner.name, sim_state=sim)
        helper.communicate_with(owner, sim_state=sim)

        self.assertIn("R_REQ", owner.mental_model["knowledge"].rules)
        blockers, _ = owner._construction_action_blockers(decision, {"project_id": project_id}, sim.environment, sim_state=sim)
        self.assertNotIn("missing_validation_rule_knowledge", blockers)
        sim.stop()

    def test_closure_repair_tkrq_is_anchored_to_missing_expected_rule(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_RULE_X"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
        owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)

        request = next((m for m in owner.generate_message(recipient_name=helper.name) if m["type"] == "TKRQ"), None)
        self.assertIsNotNone(request)
        self.assertEqual(request.get("project_id"), project_id)
        self.assertIn("R_RULE_X", request.get("missing_expected_rules", []))
        self.assertIn("rule:R_RULE_X", set(request.get("content", [])))
        sim.stop()

    def test_repeated_unchanged_closure_repair_emits_stalled_and_stops_refresh(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_STALL"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
        owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        stalled_result = owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        self.assertEqual(stalled_result[0].get("decision_action"), ExecutableActionType.REASSESS_PLAN.value)
        self.assertTrue(any(e.get("event_type") == "closure_episode_epistemic_repair_stalled" for e in sim.logger.recent_events))
        sim.stop()

    def test_closure_repair_clears_back_to_validation_when_missing_rule_resolved(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_RETURN"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        blocked = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
        owner._translate_brain_decision_to_legacy_action(blocked, sim.environment, sim_state=sim)

        owner.mental_model["knowledge"].add_rule("R_RETURN", [])
        snapshot = owner._snapshot_dik_provenance(sim_state=sim)
        sim.environment.construction.update_project_provenance(
            project_id,
            event="unit_rule_added",
            sim_time=sim.time,
            held_rule_ids=snapshot["held_rule_ids_at_build"],
        )
        reroute = owner._apply_policy_pivots(
            BrainDecision(selected_action=ExecutableActionType.COMMUNICATE, confidence=0.6),
            sim.environment,
            sim_state=sim,
            pivot_origin="local_refresh",
        )
        self.assertEqual(reroute.selected_action, ExecutableActionType.VALIDATE_CONSTRUCTION)
        owner._translate_brain_decision_to_legacy_action(reroute, sim.environment, sim_state=sim)
        self.assertTrue(any(e.get("event_type") == "closure_episode_returned_to_validation" for e in sim.logger.recent_events))
        sim.stop()

    def test_receive_message_dik_change_recomputes_project_state_and_closure_readiness(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_MSG"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        self.assertTrue(owner.project_closure_state.get("repair_mode"))

        owner.receive_message(
            {"type": "TKP", "sender": helper.name, "content": ["R_MSG"]},
            from_agent=helper.name,
            sim_state=sim,
        )

        blockers, _ = owner._construction_action_blockers(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            {"project_id": project_id},
            sim.environment,
            sim_state=sim,
        )
        self.assertNotIn("missing_expected_rule:R_MSG", blockers)
        self.assertFalse(owner.project_closure_state.get("repair_mode"))
        self.assertTrue(any(e.get("event_type") == "project_state_recomputed_after_dik_change" for e in sim.logger.recent_events))
        self.assertTrue(any(e.get("event_type") == "closure_episode_returned_to_validation" for e in sim.logger.recent_events))
        sim.stop()

    def test_recheck_commitment_message_triggers_blocker_relevant_refresh(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_PTR_REFRESH"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._update_closure_repair_state(project_id, ["missing_expected_rule:R_PTR_REFRESH"], sim.environment, sim_state=sim, origin="unit")

        with patch.object(owner, "_refresh_relevant_project_state_after_dik_change") as refresh_mock:
            owner.receive_message(
                {
                    "type": "TPS",
                    "sender": helper.name,
                    "content": {
                        "project_id": project_id,
                        "closure_blocker_signature": owner.project_closure_state.get("repair_blocker_signature"),
                        "response_category": "recheck_commitment",
                        "source_id": f"{owner.role}_Info",
                        "requested_rule_ids": ["R_PTR_REFRESH"],
                    },
                },
                from_agent=helper.name,
                sim_state=sim,
            )
        refresh_mock.assert_called_once()
        self.assertEqual(refresh_mock.call_args.kwargs.get("trigger_source"), f"message:{helper.name}:closure_signal")
        self.assertTrue(refresh_mock.call_args.kwargs.get("blocker_relevant"))
        self.assertIn(project_id, set(refresh_mock.call_args.kwargs.get("relevant_project_ids") or set()))
        sim.stop()

    def test_source_pointer_response_creates_pointed_reinspect_obligation(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_PTR"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._update_closure_repair_state(project_id, ["missing_expected_rule:R_PTR"], sim.environment, sim_state=sim, origin="unit")
        owner.receive_message(
            {
                "type": "TPS",
                "sender": helper.name,
                "content": {
                    "project_id": project_id,
                    "closure_blocker_signature": owner.project_closure_state.get("repair_blocker_signature"),
                    "response_category": "source_pointer",
                    "source_id": f"{owner.role}_Info",
                    "requested_rule_ids": ["R_PTR"],
                },
            },
            from_agent=helper.name,
            sim_state=sim,
        )
        pointer = owner._active_closure_source_pointer(project_id=project_id, sim_state=sim)
        self.assertIsNotNone(pointer)
        self.assertEqual(pointer.get("source_id"), f"{owner.role}_Info")
        with patch.object(owner, "_construction_action_blockers", return_value=(["missing_expected_rule:R_PTR"], project_id)):
            pivoted = owner._apply_policy_pivots(
                BrainDecision(selected_action=ExecutableActionType.COMMUNICATE, confidence=0.6),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
        self.assertEqual(pivoted.selected_action, ExecutableActionType.INSPECT_INFORMATION_SOURCE)
        self.assertEqual(pivoted.target_id, f"{owner.role}_Info")
        self.assertTrue(any(e.get("event_type") == "closure_source_pointer_committed" for e in sim.logger.recent_events))
        sim.stop()

    def test_simulator_readiness_reconciliation_triggers_on_dik_events(self):
        sim = SimulationState(phases=[])
        sim.logger.log_event(sim.time, "derivation_succeeded", {"agent": sim.agents[0].name, "rule_id": "R_UNIT"})
        reconciliation_events = [e for e in sim.logger.recent_events if e.get("event_type") == "readiness_reconciled"]
        self.assertTrue(reconciliation_events)
        self.assertEqual(reconciliation_events[-1]["payload_data"].get("trigger_event"), "derivation_succeeded")
        sim.stop()

    def test_exhausted_source_reinspect_is_suppressed_until_dependency_changes(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        signature = agent._readiness_dependency_signature(sim.environment, sim_state=sim)
        agent.source_exhaustion_state["Team_Info"] = {
            "exhausted": True,
            "exhausted_for_acquisition": True,
            "exhausted_dependency_signature": signature,
        }
        agent.post_inspect_handoff = {
            "pending": True,
            "source_id": "Team_Info",
            "dik_changed": False,
            "readiness_changed": False,
            "expires_at": sim.time + 2.0,
        }
        decision = agent._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertFalse(
            decision.selected_action == ExecutableActionType.INSPECT_INFORMATION_SOURCE and decision.target_id == "Team_Info"
        )
        self.assertTrue(any(e.get("event_type") == "exhausted_source_reinspect_suppressed" for e in sim.logger.recent_events))

        agent.source_exhaustion_state["Team_Info"]["exhausted_dependency_signature"] = "changed"
        decision_after_dependency_change = agent._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertIsNotNone(decision_after_dependency_change.selected_action)
        sim.stop()

    def test_closure_deadlock_recovery_reassigns_when_owner_cannot_act(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["closure_owner"] = owner.name
        project["closure_status"] = "assigned"

        def _fake_build(_sim_state, agent):
            if agent.name == owner.name:
                return SimpleNamespace(action_affordances=[])
            return SimpleNamespace(
                action_affordances=[
                    {"action_type": ExecutableActionType.VALIDATE_CONSTRUCTION.value, "target_id": project_id, "utility": 0.9}
                ]
            )

        with patch.object(sim.brain_context_builder, "build", side_effect=_fake_build):
            for _ in range(3):
                sim.logger.log_event(sim.time, "construction_ready_for_validation", {"project_id": project_id})

        self.assertTrue(
            any(
                e.get("event_type") in {"closure_reassignment_performed", "closure_reopened_for_support"}
                for e in sim.logger.recent_events
            )
        )
        sim.stop()

    def test_closure_owner_not_reassigned_during_epistemic_repair(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        helper = sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_LOCKED"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        owner.communicate_with(helper, sim_state=sim)
        self.assertEqual(sim.environment.construction.projects[project_id].get("closure_owner"), owner.name)
        sim.stop()

    def test_ready_for_validation_queues_project_state_obligation(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        queued = owner.team_plan_state.get("pending_outbound_messages", [])
        self.assertTrue(any(m.get("type") == "TPS" and m.get("content", {}).get("state_event") == "ready_for_validation" for m in queued))
        self.assertTrue(any(e.get("event_type") == "project_state_communication_obligation_queued" for e in sim.logger.recent_events))
        sim.stop()

    def test_validation_blocked_queues_project_state_obligation(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_BLOCKED"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        queued = owner.team_plan_state.get("pending_outbound_messages", [])
        self.assertTrue(any(m.get("type") == "TPS" and m.get("content", {}).get("state_event") == "validation_blocked_epistemic" for m in queued))
        sim.stop()

    def test_shared_source_inspection_does_not_auto_promote_team_validated_knowledge(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        packet = sim.environment.knowledge_packets["Team_Info"]

        before = dict(sim.team_knowledge_manager.validated_knowledge)
        before_info_count = len(agent.mental_model["information"])
        info_ids = [i.id for i in packet.get("information", [])]
        data_ids = [d.id for d in packet.get("data", [])]
        agent.absorb_packet(packet, accuracy=1.0, sim_state=sim, source_id="Team_Info")
        delta = agent._write_shared_source_to_team_knowledge(sim, "Team_Info", packet, info_ids, data_ids, [])

        self.assertEqual(delta["added"], [])
        self.assertEqual(before, sim.team_knowledge_manager.validated_knowledge)
        self.assertGreaterEqual(len(agent.mental_model["information"]), before_info_count)
        sim.stop()

    def test_post_role_packet_handoff_prefers_bounded_communication(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        teammate = sim.agents[1]
        agent.position = teammate.position = (8.0, 6.6)

        role_source = f"{agent.role}_Info"
        agent.source_memory_state.setdefault(role_source, {})["pending_role_share_ids"] = ["rule:R_REQ"]
        agent.post_inspect_handoff = {
            "pending": True,
            "source_id": role_source,
            "dik_changed": True,
            "role_packet_dik_ids": ["rule:R_REQ"],
        }
        agent.communication_state["last_meaningful_exchange_time"] = -1.0

        followup = agent._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertEqual(followup.selected_action, ExecutableActionType.COMMUNICATE)

        agent.source_memory_state[role_source]["pending_role_share_ids"] = []
        followup_after_share = agent._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertNotEqual(followup_after_share.reason_summary, "Post-inspect role-packet uptake prioritized for team DIK integration.")
        sim.stop()



    def test_post_inspect_followup_does_not_force_engineer_gap_without_local_evidence(self):
        sim = SimulationState(phases=[])
        architect = next(a for a in sim.agents if a.role == "Architect")
        engineer = next(a for a in sim.agents if a.role == "Engineer")
        architect.position = engineer.position = (8.0, 6.6)

        role_source = "Architect_Info"
        architect.post_inspect_handoff = {
            "pending": True,
            "source_id": role_source,
            "dik_changed": True,
            "role_packet_dik_ids": ["rule:R_ARCH"],
            "blockers": ["no_issue"],
        }
        architect.source_memory_state.setdefault(role_source, {})["pending_role_share_ids"] = []
        architect.source_memory_state.setdefault(role_source, {})["ever_inspected"] = True
        botanist = next(a for a in sim.agents if a.role == "Botanist")
        botanist.source_memory_state.setdefault("Botanist_Info", {})["ever_inspected"] = True
        botanist.source_inspection_state["Botanist_Info"] = "inspected"
        engineer.source_memory_state.setdefault("Engineer_Info", {})["ever_inspected"] = False
        engineer.source_inspection_state["Engineer_Info"] = "unseen"

        followup = architect._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertNotIn("cross-role witness gap", followup.reason_summary.lower())
        sim.stop()

    def test_post_inspect_followup_engineer_gap_strengthens_with_local_missing_rule_evidence(self):
        sim = SimulationState(phases=[])
        architect = next(a for a in sim.agents if a.role == "Architect")
        engineer = next(a for a in sim.agents if a.role == "Engineer")
        botanist = next(a for a in sim.agents if a.role == "Botanist")
        architect.position = engineer.position = botanist.position = (8.0, 6.6)

        architect.post_inspect_handoff = {
            "pending": True,
            "source_id": "Architect_Info",
            "dik_changed": True,
            "role_packet_dik_ids": ["rule:R_ARCH"],
            "blockers": ["missing_expected_rule:R_HOUSE_VALIDITY"],
            "expires_at": 99.0,
        }
        architect.known_gaps.add("missing_expected_rule:R_HOUSE_VALIDITY")
        architect.source_memory_state.setdefault("Architect_Info", {})["ever_inspected"] = True
        botanist.source_memory_state.setdefault("Botanist_Info", {})["ever_inspected"] = True
        botanist.source_inspection_state["Botanist_Info"] = "inspected"
        engineer.source_memory_state.setdefault("Engineer_Info", {})["ever_inspected"] = False
        engineer.source_inspection_state["Engineer_Info"] = "unseen"

        followup = architect._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
        self.assertEqual(followup.selected_action, ExecutableActionType.COMMUNICATE)
        self.assertIn("cross-role witness gap", followup.reason_summary)
        sim.stop()

    def test_communication_progress_requires_epistemic_effect(self):
        sim = SimulationState(phases=[])
        sender, receiver = sim.agents[0], sim.agents[1]
        sender.position = receiver.position = (8.0, 6.6)

        sender.known_gaps.clear()
        sender.communicate_with(receiver, sim_state=sim)
        self.assertGreater(sender.communication_state.get("no_effect_streak", 0), -1)
        self.assertFalse(sender.communication_state.get("last_exchange_effects", {}).get("meaningful", False))

        sender.mental_model["knowledge"].add_rule("R_REQ", [])
        sender.receive_message({"type": "TKRQ", "sender": receiver.name, "content": ["rule:R_REQ"]}, from_agent=receiver.name, sim_state=sim)
        outcome = sender.communicate_with(receiver, sim_state=sim)
        self.assertTrue(outcome.get("meaningful"))
        self.assertIn("R_REQ", receiver.mental_model["knowledge"].rules)
        sim.stop()

    def test_no_exact_rule_can_emit_source_pointer_without_dik_injection(self):
        sim = SimulationState(phases=[])
        requester, responder = sim.agents[0], sim.agents[1]
        requester.position = responder.position = (8.0, 6.6)
        rules_before = len(requester.mental_model["knowledge"].rules)
        responder.receive_message(
            {"type": "TKRQ", "sender": requester.name, "content": ["rule:R_GREENHOUSE_SUPPORT_DEPENDENCY"], "project_id": "Build_Table_A"},
            from_agent=requester.name,
            sim_state=sim,
        )
        queued = responder.team_plan_state.get("pending_outbound_messages", [])
        self.assertTrue(any(m.get("type") == "TPS" and m.get("content", {}).get("response_category") in {"source_pointer", "recheck_commitment", "teammate_redirect", "no_useful_response"} for m in queued))
        responder.communicate_with(requester, sim_state=sim)
        self.assertEqual(rules_before, len(requester.mental_model["knowledge"].rules))
        self.assertTrue(any(e.get("event_type") == "closure_repair_response_category" for e in sim.logger.recent_events))
        sim.stop()

    def test_closure_repair_project_scope_isolated_between_projects(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        helper = sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)
        for project_id, rule_id in [("Build_Table_A", "R_A"), ("Build_Table_B", "R_B")]:
            project = sim.environment.construction.projects[project_id]
            project["status"] = "ready_for_validation"
            project["expected_rules"] = [rule_id]
            sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)

        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment("Build_Table_A", environment=sim.environment, sim_state=sim, reason="unit")
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id="Build_Table_A", confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        request_a = next((m for m in owner.generate_message(recipient_name=helper.name) if m["type"] == "TKRQ"), {})
        self.assertEqual(request_a.get("project_id"), "Build_Table_A")
        self.assertIn("rule:R_A", set(request_a.get("content", [])))
        self.assertNotIn("rule:R_B", set(request_a.get("content", [])))

        owner._start_project_closure_commitment("Build_Table_B", environment=sim.environment, sim_state=sim, reason="unit_switch")
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id="Build_Table_B", confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        request_b = next((m for m in owner.generate_message(recipient_name=helper.name) if m["type"] == "TKRQ"), {})
        self.assertEqual(request_b.get("project_id"), "Build_Table_B")
        self.assertIn("rule:R_B", set(request_b.get("content", [])))
        self.assertNotIn("rule:R_A", set(request_b.get("content", [])))
        sim.stop()

    def test_unsatisfiable_exact_rule_request_is_deduplicated(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_MISS"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
        owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)

        owner.communicate_with(helper, sim_state=sim)
        first_sent = len([e for e in sim.logger.recent_events if e.get("event_type") == "closure_repair_request_sent"])
        owner.communicate_with(helper, sim_state=sim)
        second_sent = len([e for e in sim.logger.recent_events if e.get("event_type") == "closure_repair_request_sent"])
        chosen = [e for e in sim.logger.recent_events if e.get("event_type") == "closure_repair_strategy_chosen"]
        self.assertGreaterEqual(second_sent, first_sent)
        chosen_categories = [((e.get("payload_data", {}) or {}).get("request_category")) for e in chosen]
        self.assertTrue(any(c == "exact_rule" for c in chosen_categories))
        self.assertLessEqual(sum(1 for c in chosen_categories if c == "exact_rule"), 2)
        self.assertTrue(any(e.get("event_type") == "closure_repair_exact_rule_unsatisfiable" for e in sim.logger.recent_events))
        sim.stop()

    def test_closure_repair_response_category_is_explicit(self):
        sim = SimulationState(phases=[])
        requester, responder = sim.agents[0], sim.agents[1]
        requester.position = responder.position = (8.0, 6.6)
        responder.receive_message(
            {
                "type": "TKRQ",
                "sender": requester.name,
                "content": ["rule:R_UNKNOWN"],
                "project_id": "Build_Table_A",
                "request_modes": ["teammate_redirect"],
            },
            from_agent=requester.name,
            sim_state=sim,
        )
        queued = responder.team_plan_state.get("pending_outbound_messages", [])
        tps = next((m for m in queued if m.get("type") == "TPS"), {})
        self.assertIn(tps.get("content", {}).get("response_category"), {"teammate_redirect", "no_useful_response"})
        self.assertIsNotNone(tps.get("content", {}).get("response_category"))
        sim.stop()

    def test_source_pointer_creates_closure_scoped_inspect_obligation(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_PTR"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner.project_closure_state["repair_mode"] = True
        owner.project_closure_state["repair_blocker_signature"] = f"{project_id}|missing_expected_rule:R_PTR|R_PTR"
        owner.receive_message(
            {
                "type": "TPS",
                "sender": sim.agents[1].name,
                "content": {
                    "project_id": project_id,
                    "closure_blocker_signature": owner.project_closure_state["repair_blocker_signature"],
                    "response_category": "source_pointer",
                    "source_id": f"{owner.role}_Info",
                    "requested_rule_ids": ["R_PTR"],
                },
            },
            from_agent=sim.agents[1].name,
            sim_state=sim,
        )
        pointer = owner.project_closure_state.get("source_pointer", {})
        self.assertEqual(pointer.get("project_id"), project_id)
        self.assertEqual(pointer.get("source_id"), f"{owner.role}_Info")
        resolved_source, _ = owner._resolve_inspect_target(
            BrainDecision(selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="Team_Info", confidence=0.7),
            sim.environment,
            sim_state=sim,
        )
        self.assertEqual(resolved_source, f"{owner.role}_Info")
        sim.stop()

    def test_source_pointer_inspect_recomputes_closure_and_shrinks_signature(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_PTR_RECOMP"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner.project_closure_state["repair_mode"] = True
        owner.project_closure_state["repair_blocker_signature"] = f"{project_id}|missing_expected_rule:R_PTR_RECOMP|R_PTR_RECOMP"
        owner._set_closure_source_pointer(
            project_id=project_id,
            blocker_signature=owner.project_closure_state["repair_blocker_signature"],
            source_id=f"{owner.role}_Info",
            missing_rule_ids=["R_PTR_RECOMP"],
            sim_state=sim,
        )

        original_absorb = owner.absorb_packet

        def patched_absorb(packet, accuracy=1.0, sim_state=None, source_id=None):
            original_absorb(packet, accuracy=accuracy, sim_state=sim_state, source_id=source_id)
            owner.mental_model["knowledge"].add_rule("R_PTR_RECOMP", [])

        owner.absorb_packet = patched_absorb
        selection = owner._select_source_access_target(sim.environment, f"{owner.role}_Info", sim_state=sim)
        if selection and selection.get("position") is not None:
            owner.position = tuple(selection["position"])
        owner._inspect_source(sim.environment, f"{owner.role}_Info", sim_state=sim)

        self.assertTrue(any(e.get("event_type") == "project_state_recomputed_after_dik_change" for e in sim.logger.recent_events))
        self.assertNotIn("R_PTR_RECOMP", str(owner.project_closure_state.get("repair_blocker_signature") or ""))
        sim.stop()

    def test_closure_owner_pointed_obligation_is_not_demoted_to_generic_communication(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_OWNER_PTR"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner.project_closure_state["repair_mode"] = True
        owner.project_closure_state["repair_blocker_signature"] = f"{project_id}|missing_expected_rule:R_OWNER_PTR|R_OWNER_PTR"
        owner._set_closure_source_pointer(
            project_id=project_id,
            blocker_signature=owner.project_closure_state["repair_blocker_signature"],
            source_id=f"{owner.role}_Info",
            missing_rule_ids=["R_OWNER_PTR"],
            sim_state=sim,
        )
        with patch.object(owner, "_construction_action_blockers", return_value=(["missing_expected_rule:R_OWNER_PTR", "missing_validation_rule_knowledge"], project_id)), patch.object(owner, "_classify_validation_blockers", return_value={"epistemic_blockers": ["missing_expected_rule:R_OWNER_PTR", "missing_validation_rule_knowledge"], "non_epistemic_blockers": []}), patch.object(owner, "_can_attempt_verbal_plan_communication", return_value=True):
            rewritten = owner._apply_policy_pivots(
                BrainDecision(selected_action=ExecutableActionType.COMMUNICATE, confidence=0.6),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
        self.assertNotEqual(rewritten.selected_action, ExecutableActionType.COMMUNICATE)
        if rewritten.selected_action == ExecutableActionType.INSPECT_INFORMATION_SOURCE:
            self.assertEqual(rewritten.target_id, f"{owner.role}_Info")
        sim.stop()

    def test_blocker_relevant_refresh_uses_project_local_stale_deferral(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        sim.environment.construction.projects[project_id]["status"] = "ready_for_validation"
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner.project_closure_state["last_blocker_relevant_refresh"] = {
            "project_id": project_id,
            "source_id": f"{owner.role}_Info",
            "trigger_source": "unit",
            "expires_at": sim.time + 5.0,
        }
        with patch.object(owner, "_epistemic_sufficiency", return_value={"stale_grounding": True, "role_missing": False}):
            blockers = owner._build_readiness_blockers(sim.environment, sim_state=sim)
        self.assertIn("stale_epistemic_grounding_project_deferred", blockers)
        self.assertNotIn("stale_epistemic_grounding", blockers)
        sim.stop()

    def test_named_missing_rule_binding_emits_precursor_after_recompute(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_GREENHOUSE_SUPPORT_DEPENDENCY"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._update_closure_repair_state(
            project_id,
            ["missing_expected_rule:R_GREENHOUSE_SUPPORT_DEPENDENCY", "missing_validation_rule_knowledge"],
            sim.environment,
            sim_state=sim,
            origin="unit",
        )
        owner._recompute_project_state_after_dik_change(
            sim_state=sim,
            trigger_source="unit_recompute",
            changed_information_ids={"some_new_fact"},
            blocker_relevant=True,
        )
        recompute_event = next(
            (e for e in reversed(sim.logger.recent_events) if e.get("event_type") == "project_state_recomputed_after_dik_change"),
            {},
        )
        payload = recompute_event.get("payload_data", {}) or {}
        self.assertIn(payload.get("closure_named_rule_binding_status"), {"shrunk", "unchanged"})
        self.assertTrue(
            any(
                str(b).startswith("missing_expected_rule_precursor:R_GREENHOUSE_SUPPORT_DEPENDENCY:")
                for b in payload.get("closure_blockers", [])
            )
        )
        sim.stop()

    def test_nonproductive_pointed_source_exhaustion_blocks_recommit_for_same_signature(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_EXH"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        signature = f"{project_id}|missing_expected_rule:R_EXH|R_EXH"
        owner.project_closure_state["repair_mode"] = True
        owner.project_closure_state["repair_blocker_signature"] = signature
        ex_key = owner._closure_pointer_exhaustion_key(project_id, signature, f"{owner.role}_Info")
        owner.project_closure_state.setdefault("source_pointer_exhaustion", {})[ex_key] = {
            "exhausted": True,
            "reason": "nonproductive_inspect_attempts",
            "attempt_count": 2,
            "source_id": f"{owner.role}_Info",
            "project_id": project_id,
        }
        committed = owner._set_closure_source_pointer(
            project_id=project_id,
            blocker_signature=signature,
            source_id=f"{owner.role}_Info",
            missing_rule_ids=["R_EXH"],
            sim_state=sim,
        )
        self.assertFalse(committed)
        self.assertTrue(any(e.get("event_type") == "pointed_source_obligation_deferred" for e in sim.logger.recent_events))
        sim.stop()

    def test_pointed_inspect_passes_blocker_relevant_project_hint_to_epistemic_pipeline(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_PTR_HINT"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._update_closure_repair_state(project_id, ["missing_expected_rule:R_PTR_HINT"], sim.environment, sim_state=sim, origin="unit")
        owner._set_closure_source_pointer(
            project_id=project_id,
            blocker_signature=owner.project_closure_state.get("repair_blocker_signature"),
            source_id=f"{owner.role}_Info",
            missing_rule_ids=["R_PTR_HINT"],
            sim_state=sim,
        )
        selection = owner._select_source_access_target(sim.environment, f"{owner.role}_Info", sim_state=sim)
        if selection and selection.get("position") is not None:
            owner.position = tuple(selection["position"])
        with patch.object(owner, "_trigger_epistemic_update_pipeline") as trigger_mock:
            owner._inspect_source(sim.environment, f"{owner.role}_Info", sim_state=sim)
        self.assertGreaterEqual(trigger_mock.call_count, 1)
        relevant_call = trigger_mock.call_args_list[-1]
        self.assertTrue(relevant_call.kwargs.get("blocker_relevant"))
        self.assertIn(project_id, set(relevant_call.kwargs.get("relevant_project_ids") or set()))
        self.assertEqual(
            relevant_call.kwargs.get("trigger_source"),
            f"closure_pointed_source_followthrough:{project_id}:{owner.role}_Info",
        )
        sim.stop()

    def test_live_closure_missing_rules_shrink_without_new_provenance_system(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_LIVE"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        decision = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)
        owner._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        self.assertIn("R_LIVE", owner._closure_repair_missing_rules(project_id, sim.environment))
        owner.mental_model["knowledge"].add_rule("R_LIVE", [])
        owner._recompute_project_state_after_dik_change(sim_state=sim, trigger_source="unit", changed_rule_ids={"R_LIVE"})
        self.assertNotIn("R_LIVE", owner._closure_repair_missing_rules(project_id, sim.environment))
        sim.stop()

    def test_closure_blocker_shrink_returns_to_validation_via_canonical_path(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["correct"] = True
        project["expected_rules"] = ["R_CANON"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        owner.receive_message(
            {"type": "TKP", "sender": sim.agents[1].name, "content": ["R_CANON"]},
            from_agent=sim.agents[1].name,
            sim_state=sim,
        )
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        reroute = owner._apply_policy_pivots(
            BrainDecision(selected_action=ExecutableActionType.COMMUNICATE, target_id=project_id, confidence=0.6),
            sim.environment,
            sim_state=sim,
            pivot_origin="unit",
        )
        self.assertEqual(reroute.selected_action, ExecutableActionType.VALIDATE_CONSTRUCTION)
        self.assertTrue(any(e.get("event_type") == "closure_episode_returned_to_validation" for e in sim.logger.recent_events))
        sim.stop()

    def test_closure_support_focus_is_bounded_when_unchanged(self):
        sim = SimulationState(phases=[])
        helper = sim.agents[1]
        project = sim.environment.construction.projects["Build_Table_A"]
        project["status"] = "needs_repair"
        for _ in range(3):
            helper._apply_policy_pivots(
                BrainDecision(selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="Team_Info", confidence=0.7),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
        self.assertTrue(any(e.get("event_type") == "closure_support_focus_skipped_exhausted" for e in sim.logger.recent_events))
        sim.stop()

    def test_closure_repair_category_escalates_then_exhausts(self):
        sim = SimulationState(phases=[])
        owner, helper = sim.agents[0], sim.agents[1]
        owner.position = helper.position = (8.0, 6.6)
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_NEVER"]
        sim.environment.construction.update_project_provenance(project_id, event="unit_setup", held_rule_ids=[], sim_time=sim.time)
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._translate_brain_decision_to_legacy_action(
            BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9),
            sim.environment,
            sim_state=sim,
        )
        for _ in range(8):
            owner.communicate_with(helper, sim_state=sim)
            sim.time += 3.5
        chosen = [e for e in sim.logger.recent_events if e.get("event_type") == "closure_repair_strategy_chosen"]
        categories = set()
        for e in chosen:
            category = (e.get("payload_data", {}) or {}).get("request_category")
            if category:
                categories.add(category)
        self.assertTrue({"exact_rule", "precursor_info", "source_pointer"}.intersection(categories))
        self.assertTrue(any(e.get("event_type") == "closure_repair_response_category" for e in sim.logger.recent_events))
        sim.stop()

    def test_no_useful_response_exhausts_strategy_for_signature(self):
        sim = SimulationState(phases=[])
        owner = sim.agents[0]
        project_id = "Build_Table_A"
        project = sim.environment.construction.projects[project_id]
        project["status"] = "ready_for_validation"
        project["expected_rules"] = ["R_NONE"]
        self._prime_readiness_without_rules(sim, owner)
        owner._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        owner._update_closure_repair_state(project_id, ["missing_expected_rule:R_NONE"], sim.environment, sim_state=sim, origin="unit")
        signature = owner.project_closure_state.get("repair_blocker_signature")
        sender = sim.agents[1].name
        owner.receive_message(
            {
                "type": "TPS",
                "sender": sender,
                "content": {
                    "project_id": project_id,
                    "closure_blocker_signature": signature,
                    "response_category": "no_useful_response",
                    "requested_rule_ids": ["R_NONE"],
                },
            },
            from_agent=sender,
            sim_state=sim,
        )
        bucket = owner._closure_repair_retry_bucket(project_id, signature)
        failed = set(bucket.get("failed_categories", {}).get(sender, []))
        self.assertTrue(bucket.get("exhausted"))
        self.assertTrue({"exact_rule", "precursor_info", "source_pointer", "recheck_commitment", "teammate_redirect"}.issubset(failed))
        sim.stop()

    def test_recheck_commitment_creates_bounded_reinspect_next_step_for_responder(self):
        sim = SimulationState(phases=[])
        requester, responder = sim.agents[0], sim.agents[1]
        project_id = "Build_Table_A"
        sim.environment.construction.projects[project_id]["status"] = "ready_for_validation"
        sim.environment.construction.projects[project_id]["expected_rules"] = ["R_RECHECK"]
        self._prime_readiness_without_rules(sim, responder)
        responder._start_project_closure_commitment(project_id, environment=sim.environment, sim_state=sim, reason="unit")
        responder._update_closure_repair_state(project_id, ["missing_expected_rule:R_RECHECK"], sim.environment, sim_state=sim, origin="unit")
        responder.receive_message(
            {
                "type": "TKRQ",
                "sender": requester.name,
                "project_id": project_id,
                "closure_blocker_signature": responder.project_closure_state.get("repair_blocker_signature"),
                "request_modes": ["recheck_commitment"],
                "content": ["rule:R_RECHECK"],
            },
            from_agent=requester.name,
            sim_state=sim,
        )
        pending = dict(responder.communication_state.get("closure_recheck_pending") or {})
        self.assertEqual(pending.get("recipient"), requester.name)
        self.assertEqual(pending.get("source_id"), f"{responder.role}_Info")
        self.assertEqual(responder.progress_tracker.get("forced_pivot"), ExecutableActionType.INSPECT_INFORMATION_SOURCE.value)
        sim.stop()

    def test_blocked_project_support_pressure_reroutes_unrelated_inspect(self):
        sim = SimulationState(phases=[])
        helper = sim.agents[1]
        project = sim.environment.construction.projects["Build_Table_A"]
        project["status"] = "needs_repair"
        rewritten = helper._apply_policy_pivots(
            BrainDecision(selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="Team_Info", confidence=0.7),
            sim.environment,
            sim_state=sim,
            pivot_origin="unit",
        )
        self.assertIn(rewritten.selected_action, {ExecutableActionType.COMMUNICATE, ExecutableActionType.WAIT})
        self.assertTrue(any(e.get("event_type") == "blocked_project_support_pressure_applied" for e in sim.logger.recent_events))
        sim.stop()


if __name__ == "__main__":
    unittest.main()
