import unittest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class SupportDependencyChainPatchTests(unittest.TestCase):
    def _complete_project_structure(self, cm, project_id, *, actor="Architect", sim_time=1.0):
        project = cm.projects[project_id]
        cm.deliver_resource(project_id, "bricks", quantity=int(project["required_resources"]["bricks"]))
        for _ in range(len(project.get("build_steps") or [])):
            cm.execute_build_step(project_id, actor=actor, sim_time=sim_time)

    def _complete_house(self, cm, house_id):
        self._complete_project_structure(cm, house_id, sim_time=1.0)
        cm.record_project_epistemic_externalization(house_id, entry_type="claim", note="claim", actor="Architect", sim_time=1.1)
        cm.record_project_epistemic_externalization(house_id, entry_type="evidence", note="evidence", actor="Architect", sim_time=1.2)
        cm.record_project_epistemic_externalization(house_id, entry_type="design_note", note="design", actor="Architect", sim_time=1.3)

    def test_missing_food_support_creates_or_selects_provider(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            house_id, _ = cm.create_project("site_b", structure_type="house")
            self._complete_house(cm, house_id)

            rewritten = agent._resolve_support_deficit_decision(
                BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=house_id, confidence=0.7),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
            self.assertIsNotNone(rewritten)
            self.assertEqual(rewritten.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)
            provider = cm.projects.get(str(rewritten.target_id))
            self.assertEqual(provider.get("type"), "greenhouse")
        finally:
            sim.stop()

    def test_provider_complete_creates_or_selects_connector(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            house_id, _ = cm.create_project("site_b", structure_type="house")
            self._complete_house(cm, house_id)
            provider_id, _ = cm.create_project("site_b", structure_type="greenhouse")
            self._complete_project_structure(cm, provider_id, sim_time=2.0)

            rewritten = agent._resolve_support_deficit_decision(
                BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=house_id, confidence=0.7),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
            self.assertIsNotNone(rewritten)
            self.assertEqual(rewritten.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)
            connector = cm.projects.get(str(rewritten.target_id))
            self.assertEqual(connector.get("type"), "food_connector")
        finally:
            sim.stop()

    def test_connector_missing_materials_routes_transport(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            house_id, _ = cm.create_project("site_b", structure_type="house")
            self._complete_house(cm, house_id)
            provider_id, _ = cm.create_project("site_b", structure_type="greenhouse")
            self._complete_project_structure(cm, provider_id, sim_time=2.0)
            connector_id, _ = cm.create_project("site_b", structure_type="food_connector")

            rewritten = agent._resolve_support_deficit_decision(
                BrainDecision(selected_action=ExecutableActionType.WAIT, target_id=provider_id, confidence=0.6),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
            self.assertEqual(rewritten.selected_action, ExecutableActionType.TRANSPORT_RESOURCES)
            self.assertEqual(rewritten.target_id, connector_id)
        finally:
            sim.stop()

    def test_connector_material_ready_routes_construction(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            house_id, _ = cm.create_project("site_b", structure_type="house")
            self._complete_house(cm, house_id)
            provider_id, _ = cm.create_project("site_b", structure_type="greenhouse")
            self._complete_project_structure(cm, provider_id, sim_time=2.0)
            connector_id, _ = cm.create_project("site_b", structure_type="food_connector")
            connector = cm.projects[connector_id]
            cm.deliver_resource(connector_id, "bricks", quantity=int(connector["required_resources"]["bricks"]))

            rewritten = agent._resolve_support_deficit_decision(
                BrainDecision(selected_action=ExecutableActionType.WAIT, target_id=connector_id, confidence=0.6),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
            self.assertEqual(rewritten.selected_action, ExecutableActionType.START_CONSTRUCTION)
            self.assertEqual(rewritten.target_id, connector_id)
        finally:
            sim.stop()

    def test_connector_complete_attempts_attachment_and_updates_support(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            house_id, _ = cm.create_project("site_b", structure_type="house")
            self._complete_house(cm, house_id)
            provider_id, _ = cm.create_project("site_b", structure_type="greenhouse")
            connector_id, _ = cm.create_project("site_b", structure_type="food_connector")
            self._complete_project_structure(cm, provider_id, sim_time=2.0)
            self._complete_project_structure(cm, connector_id, sim_time=2.1)

            rewritten = agent._resolve_support_deficit_decision(
                BrainDecision(selected_action=ExecutableActionType.WAIT, target_id=house_id, confidence=0.6),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
            self.assertIsNotNone(rewritten)
            summary = cm.get_structure_support_summary(house_id)
            self.assertGreaterEqual(summary["counts"].get("food", 0), 1)
            self.assertTrue(any(e.get("event_type") == "connector_attachment_attempted" for e in sim.logger.get_recent_events(300)))
        finally:
            sim.stop()

    def test_final_support_resolution_returns_house_validation(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            house_id, _ = cm.create_project("site_b", structure_type="house")
            house = cm.projects[house_id]
            house["support_requirements"] = {"food": 1}
            house["support_counts"] = {"food": 0}
            self._complete_house(cm, house_id)
            provider_id, _ = cm.create_project("site_b", structure_type="greenhouse")
            connector_id, _ = cm.create_project("site_b", structure_type="food_connector")
            self._complete_project_structure(cm, provider_id, sim_time=2.0)
            self._complete_project_structure(cm, connector_id, sim_time=2.1)

            rewritten = agent._resolve_support_deficit_decision(
                BrainDecision(selected_action=ExecutableActionType.WAIT, target_id=house_id, confidence=0.6),
                sim.environment,
                sim_state=sim,
                pivot_origin="unit",
            )
            self.assertIsNotNone(rewritten)
            self.assertEqual(rewritten.selected_action, ExecutableActionType.VALIDATE_CONSTRUCTION)
            self.assertEqual(rewritten.target_id, house_id)
        finally:
            sim.stop()

    def test_support_deficit_chain_preempts_repair(self):
        sim = SimulationState(phases=[], flash_mode=True)
        try:
            agent = sim.agents[0]
            cm = sim.environment.construction
            house_id, _ = cm.create_project("site_b", structure_type="house")
            self._complete_house(cm, house_id)
            decision = BrainDecision(selected_action=ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, target_id=house_id, confidence=0.7)
            blockers, resolved_project_id = agent._construction_action_blockers(
                decision,
                {"type": "idle", "decision_action": ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value, "project_id": house_id},
                environment=sim.environment,
                sim_state=sim,
            )
            self.assertEqual(resolved_project_id, house_id)
            self.assertIn("support_dependency_chain_active", blockers)
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
