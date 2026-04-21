import unittest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class ConstructionEpistemicWorkspaceTests(unittest.TestCase):
    def _prime_agent(self, sim, agent):
        team_packet = sim.environment.knowledge_packets["Team_Info"]
        role_packet = sim.environment.knowledge_packets.get(f"{agent.role}_Info", {})
        agent.mental_model["information"].add(team_packet["information"][0])
        agent.mental_model["information"].add(team_packet["information"][1])
        if role_packet.get("information"):
            agent.mental_model["information"].add(role_packet["information"][0])
        agent.mental_model["knowledge"].rules.append("R_HOUSE_VALIDITY")
        for src in ["Team_Info", f"{agent.role}_Info", "Engineer_Info"]:
            agent.source_inspection_state[src] = "inspected"
            memory = agent.source_memory_state.setdefault(src, {})
            memory["ever_inspected"] = True
            memory["last_inspected_time"] = float(sim.time)
            memory["last_verified_time"] = float(sim.time)
            memory["memory_confidence"] = 0.95

    def test_delivery_stages_materials_without_physical_completion(self):
        sim = SimulationState(phases=[])
        project = sim.environment.construction.projects["Build_Table_B"]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource("Build_Table_B", "bricks", quantity=required)
        self.assertTrue(project["build_ready"])
        self.assertFalse(project["structurally_complete"])
        self.assertNotEqual(project["status"], "ready_for_validation")
        sim.stop()

    def test_build_step_advances_physical_state(self):
        sim = SimulationState(phases=[])
        project = sim.environment.construction.projects["Build_Table_B"]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource("Build_Table_B", "bricks", quantity=required)
        before = sum(1 for step in project["build_steps"] if step["completed"])
        ok, reason, _step = sim.environment.construction.execute_build_step("Build_Table_B", actor="Architect", sim_time=1.0)
        after = sum(1 for step in project["build_steps"] if step["completed"])
        self.assertTrue(ok, reason)
        self.assertGreater(after, before)
        sim.stop()

    def test_epistemic_externalization_updates_workspace(self):
        sim = SimulationState(phases=[])
        ok = sim.environment.construction.record_project_epistemic_externalization(
            "Build_Table_B",
            entry_type="claim",
            note="Housing shell can satisfy pressure constraints.",
            references=["R_HOUSE_VALIDITY"],
            actor="Architect",
            sim_time=1.0,
        )
        project = sim.environment.construction.projects["Build_Table_B"]
        self.assertTrue(ok)
        self.assertTrue(project["epistemic_workspace"]["entries"])
        self.assertEqual(project["epistemic_workspace"]["entries"][-1]["entry_type"], "claim")
        sim.stop()

    def test_validation_requires_epistemic_and_physical_completeness(self):
        sim = SimulationState(phases=[])
        project = sim.environment.construction.projects["Build_Table_B"]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource("Build_Table_B", "bricks", quantity=required)
        sim.environment.construction.mark_validated("Build_Table_B", is_valid=True, actor="Architect", sim_time=2.0)
        self.assertNotEqual(project["status"], "complete")

        for _ in range(len(project["build_steps"])):
            sim.environment.construction.execute_build_step("Build_Table_B", actor="Architect", sim_time=3.0)
        sim.environment.construction.record_project_epistemic_externalization("Build_Table_B", entry_type="claim", note="claim", actor="Architect", sim_time=4.0)
        sim.environment.construction.record_project_epistemic_externalization("Build_Table_B", entry_type="evidence", note="evidence", actor="Architect", sim_time=4.1)
        sim.environment.construction.record_project_epistemic_externalization("Build_Table_B", entry_type="design_note", note="layout", actor="Architect", sim_time=4.2)
        sim.environment.construction.mark_validated("Build_Table_B", is_valid=True, actor="Architect", sim_time=5.0)
        self.assertEqual(project["status"], "complete")
        sim.stop()

    def test_agent_selects_build_after_staging_and_epistemic_support(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_agent(sim, agent)
        project = sim.environment.construction.projects["Build_Table_B"]
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource("Build_Table_B", "bricks", quantity=required)
        sim.environment.construction.record_project_epistemic_externalization("Build_Table_B", entry_type="claim", note="claim", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization("Build_Table_B", entry_type="evidence", note="evidence", actor=agent.name, sim_time=sim.time)
        decision = BrainDecision(selected_action=ExecutableActionType.START_CONSTRUCTION, target_id="Build_Table_B", confidence=0.9)
        translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        self.assertEqual(translated[0]["type"], "construct")
        sim.stop()

    def test_transport_unbound_target_is_suppressed(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        agent.active_actions = [{
            "type": "transport_resources",
            "duration": 30.0,
            "progress": 0.0,
            "decision_action": ExecutableActionType.TRANSPORT_RESOURCES.value,
            "project_id": None,
        }]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, dt=0.1)
        self.assertTrue(any(e.get("event_type") == "transport_suppressed_stale_or_unbound" for e in sim.logger.recent_events))
        sim.stop()

    def test_bridge_path_remains_concrete_construction(self):
        sim = SimulationState(phases=[])
        bridge = sim.environment.construction.bridges["bridge_bc"]
        self.assertEqual(bridge.status, "not_started")
        self.assertTrue(sim.environment.construction.build_bridge_bc(quantity=bridge.required_resources))
        self.assertEqual(bridge.status, "complete")
        self.assertTrue(sim.environment.is_interaction_target_unlocked("Build_Table_C"))
        sim.stop()


if __name__ == "__main__":
    unittest.main()
