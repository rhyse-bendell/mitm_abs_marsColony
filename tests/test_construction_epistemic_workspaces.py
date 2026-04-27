import unittest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class ConstructionEpistemicWorkspaceTests(unittest.TestCase):
    def _ensure_project(self, sim, target_id="Build_Site_B"):
        project_id = sim.environment.construction.resolve_project_id(target_id, create_if_missing=True)
        return project_id, sim.environment.construction.projects[project_id]

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
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        self.assertTrue(project["build_ready"])
        self.assertFalse(project["structurally_complete"])
        self.assertNotEqual(project["status"], "ready_for_validation")
        sim.stop()

    def test_build_step_advances_physical_state(self):
        sim = SimulationState(phases=[])
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        before = sum(1 for step in project["build_steps"] if step["completed"])
        before_workspace = len(project["epistemic_workspace"]["entries"])
        before_timeline = len(project["provenance"]["timeline"])
        ok, reason, _step = sim.environment.construction.execute_build_step(project_id, actor="Architect", sim_time=1.0)
        after = sum(1 for step in project["build_steps"] if step["completed"])
        self.assertTrue(ok, reason)
        self.assertGreater(after, before)
        self.assertGreater(len(project["epistemic_workspace"]["entries"]), before_workspace)
        self.assertEqual(project["epistemic_workspace"]["entries"][-1]["entry_type"], "design_note")
        self.assertGreater(len(project["provenance"]["timeline"]), before_timeline)
        sim.stop()

    def test_epistemic_externalization_updates_workspace(self):
        sim = SimulationState(phases=[])
        ok = sim.environment.construction.record_project_epistemic_externalization(
            "Build_Site_B",
            entry_type="claim",
            note="Housing shell can satisfy pressure constraints.",
            references=["R_HOUSE_VALIDITY"],
            actor="Architect",
            sim_time=1.0,
        )
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        self.assertTrue(ok)
        self.assertTrue(project["epistemic_workspace"]["entries"])
        self.assertEqual(project["epistemic_workspace"]["entries"][-1]["entry_type"], "claim")
        sim.stop()

    def test_validation_requires_epistemic_and_physical_completeness(self):
        sim = SimulationState(phases=[])
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        sim.environment.construction.mark_validated(project_id, is_valid=True, actor="Architect", sim_time=2.0)
        self.assertNotEqual(project["status"], "complete")

        for _ in range(len(project["build_steps"])):
            sim.environment.construction.execute_build_step(project_id, actor="Architect", sim_time=3.0)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="claim", note="claim", actor="Architect", sim_time=4.0)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="evidence", note="evidence", actor="Architect", sim_time=4.1)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="design_note", note="layout", actor="Architect", sim_time=4.2)
        project["support_requirements"] = {}
        project["support_counts"] = {}
        project["support_status"] = {}
        sim.environment.construction.recompute_support_status(project_id)
        sim.environment.construction.mark_validated(project_id, is_valid=True, actor="Architect", sim_time=5.0)
        self.assertEqual(project["status"], "complete")
        sim.stop()

    def test_agent_selects_build_after_staging_and_epistemic_support(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        self._prime_agent(sim, agent)
        project_id, project = self._ensure_project(sim, "Build_Site_B")
        required = int(project["required_resources"]["bricks"])
        sim.environment.construction.deliver_resource(project_id, "bricks", quantity=required)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="claim", note="claim", actor=agent.name, sim_time=sim.time)
        sim.environment.construction.record_project_epistemic_externalization(project_id, entry_type="evidence", note="evidence", actor=agent.name, sim_time=sim.time)
        decision = BrainDecision(selected_action=ExecutableActionType.START_CONSTRUCTION, target_id=project_id, confidence=0.9)
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

    def test_connector_templates_are_buildable_with_four_bricks(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        food_template = cm._template_for_structure_type("food_connector")
        water_template = cm._template_for_structure_type("water_connector")
        self.assertEqual(int(food_template["required"]), 4)
        self.assertEqual(int(water_template["required"]), 4)
        self.assertTrue(food_template["artifact_type"].startswith("food_connector"))
        self.assertTrue(water_template["artifact_type"].startswith("water_connector"))
        sim.stop()

    def test_house_support_gates_validation_completion(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        house_id, house = self._ensure_project(sim, "Build_Site_B")
        required = int(house["required_resources"]["bricks"])
        cm.deliver_resource(house_id, "bricks", quantity=required)
        for _ in range(len(house["build_steps"])):
            cm.execute_build_step(house_id, actor="Architect", sim_time=3.0)
        cm.record_project_epistemic_externalization(house_id, entry_type="claim", note="claim", actor="Architect", sim_time=4.0)
        cm.record_project_epistemic_externalization(house_id, entry_type="evidence", note="evidence", actor="Architect", sim_time=4.1)
        self.assertTrue(house["structurally_complete"])
        self.assertFalse(house["functional_support_complete"])
        cm.mark_validated(house_id, is_valid=True, actor="Architect", sim_time=5.0)
        self.assertFalse(house["validated_complete"])
        self.assertNotEqual(house["status"], "complete")
        sim.stop()

    def test_attach_connector_updates_house_support_counts(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        house_id, _ = cm.create_project("site_b", structure_type="house")
        greenhouse_id, _ = cm.create_project("site_b", structure_type="greenhouse")
        water_id, _ = cm.create_project("site_b", structure_type="water_generator")
        food_connector_id, _ = cm.create_project("site_b", structure_type="food_connector")
        water_connector_id, _ = cm.create_project("site_b", structure_type="water_connector")
        ok_food, _ = cm.attach_connector(food_connector_id, from_project_id=house_id, to_project_id=greenhouse_id, actor="Architect", sim_time=1.0)
        ok_water, _ = cm.attach_connector(water_connector_id, from_project_id=house_id, to_project_id=water_id, actor="Architect", sim_time=1.1)
        self.assertTrue(ok_food)
        self.assertTrue(ok_water)
        summary = cm.get_structure_support_summary(house_id)
        self.assertGreaterEqual(summary["counts"].get("food", 0), 1)
        self.assertGreaterEqual(summary["counts"].get("water", 0), 1)
        self.assertTrue(summary["functional_support_complete"])
        sim.stop()

    def test_missing_support_requirements_reports_food_and_water_deficits(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        house_id, _ = cm.create_project("site_b", structure_type="house")
        missing = cm.get_missing_support_requirements(house_id)
        self.assertEqual(missing.get("food"), 1)
        self.assertEqual(missing.get("water"), 1)
        sim.stop()

    def test_support_deficit_produces_connector_project_need(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        house_id, _ = cm.create_project("site_b", structure_type="house")
        missing = cm.get_missing_support_requirements(house_id)
        self.assertIn("water", missing)
        connector_id = cm.find_attachable_connector(house_id, "water")
        self.assertIsNone(connector_id)
        created_id, reason = cm.find_or_create_connector_project("site_b", "water", author="test")
        self.assertTrue(created_id)
        self.assertIn(reason, {"created", "exists"})
        sim.stop()

    def test_food_connector_chain_can_resolve_house_support(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        house_id, _ = cm.create_project("site_b", structure_type="house")
        provider_id, _ = cm.create_project("site_b", structure_type="greenhouse")
        connector_id, _ = cm.create_project("site_b", structure_type="food_connector")
        for project_id in [provider_id, connector_id]:
            project = cm.projects[project_id]
            cm.deliver_resource(project_id, "bricks", quantity=int(project["required_resources"]["bricks"]))
            for _ in range(len(project["build_steps"])):
                cm.execute_build_step(project_id, actor="Architect", sim_time=2.0)
        ok, reason = cm.attach_connector(connector_id, from_project_id=house_id, to_project_id=provider_id, actor="Architect", sim_time=3.0)
        self.assertTrue(ok, reason)
        summary = cm.get_structure_support_summary(house_id)
        self.assertGreaterEqual(summary["counts"].get("food", 0), 1)
        sim.stop()

    def test_water_connector_chain_can_resolve_house_support(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        house_id, _ = cm.create_project("site_b", structure_type="house")
        provider_id, _ = cm.create_project("site_b", structure_type="water_generator")
        connector_id, _ = cm.create_project("site_b", structure_type="water_connector")
        for project_id in [provider_id, connector_id]:
            project = cm.projects[project_id]
            cm.deliver_resource(project_id, "bricks", quantity=int(project["required_resources"]["bricks"]))
            for _ in range(len(project["build_steps"])):
                cm.execute_build_step(project_id, actor="Architect", sim_time=2.0)
        ok, reason = cm.attach_connector(connector_id, from_project_id=house_id, to_project_id=provider_id, actor="Architect", sim_time=3.0)
        self.assertTrue(ok, reason)
        summary = cm.get_structure_support_summary(house_id)
        self.assertGreaterEqual(summary["counts"].get("water", 0), 1)
        sim.stop()

    def test_house_can_validate_after_support_connectors_attached(self):
        sim = SimulationState(phases=[])
        cm = sim.environment.construction
        house_id, _ = cm.create_project("site_b", structure_type="house")
        house = cm.projects[house_id]
        cm.deliver_resource(house_id, "bricks", quantity=int(house["required_resources"]["bricks"]))
        for _ in range(len(house["build_steps"])):
            cm.execute_build_step(house_id, actor="Architect", sim_time=1.0)
        cm.record_project_epistemic_externalization(house_id, entry_type="claim", note="claim", actor="Architect", sim_time=1.1)
        cm.record_project_epistemic_externalization(house_id, entry_type="evidence", note="evidence", actor="Architect", sim_time=1.2)
        cm.record_project_epistemic_externalization(house_id, entry_type="design_note", note="design", actor="Architect", sim_time=1.3)

        greenhouse_id, _ = cm.create_project("site_b", structure_type="greenhouse")
        food_connector_id, _ = cm.create_project("site_b", structure_type="food_connector")
        water_id, _ = cm.create_project("site_b", structure_type="water_generator")
        water_connector_id, _ = cm.create_project("site_b", structure_type="water_connector")
        for project_id in [greenhouse_id, food_connector_id, water_id, water_connector_id]:
            project = cm.projects[project_id]
            cm.deliver_resource(project_id, "bricks", quantity=int(project["required_resources"]["bricks"]))
            for _ in range(len(project["build_steps"])):
                cm.execute_build_step(project_id, actor="Architect", sim_time=2.0)
        self.assertTrue(cm.attach_connector(food_connector_id, from_project_id=house_id, to_project_id=greenhouse_id, actor="Architect", sim_time=3.0)[0])
        self.assertTrue(cm.attach_connector(water_connector_id, from_project_id=house_id, to_project_id=water_id, actor="Architect", sim_time=3.1)[0])
        self.assertTrue(cm.get_structure_support_summary(house_id)["functional_support_complete"])
        cm.mark_validated(house_id, is_valid=True, actor="Architect", sim_time=4.0)
        self.assertTrue(house["validated_complete"])
        self.assertEqual(house["status"], "complete")
        sim.stop()


if __name__ == "__main__":
    unittest.main()
