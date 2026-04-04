import unittest

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

    def test_shared_source_inspection_does_not_auto_promote_team_validated_knowledge(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        packet = sim.environment.knowledge_packets["Team_Info"]

        before = dict(sim.team_knowledge_manager.validated_knowledge)
        info_ids = [i.id for i in packet.get("information", [])]
        data_ids = [d.id for d in packet.get("data", [])]
        agent.absorb_packet(packet, accuracy=1.0, sim_state=sim, source_id="Team_Info")
        delta = agent._write_shared_source_to_team_knowledge(sim, "Team_Info", packet, info_ids, data_ids, [])

        self.assertEqual(delta["added"], [])
        self.assertEqual(before, sim.team_knowledge_manager.validated_knowledge)
        self.assertGreaterEqual(len(agent.mental_model["information"]), len(packet.get("information", [])))
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


if __name__ == "__main__":
    unittest.main()
