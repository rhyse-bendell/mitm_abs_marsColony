import json
import unittest
from pathlib import Path

from modules.construction import ConstructionManager
from modules.simulation import SimulationState


class ConstructionSystemRuleTests(unittest.TestCase):
    def test_site_c_blocked_until_bridge_complete(self):
        manager = ConstructionManager()
        project_id, status = manager.create_project(site_id="site_c", structure_type="house")
        self.assertEqual(status, "created")
        ok = manager.deliver_resource(project_id, "bricks", quantity=1)
        self.assertFalse(ok)
        self.assertFalse(manager.projects[project_id]["started"])

    def test_bridge_build_enables_site_c(self):
        manager = ConstructionManager()
        self.assertEqual(manager.bridges["bridge_bc"].status, "not_started")
        self.assertFalse(manager.build_bridge_bc(quantity=10))
        self.assertEqual(manager.bridges["bridge_bc"].status, "in_progress")
        self.assertTrue(manager.build_bridge_bc(quantity=10))
        self.assertEqual(manager.bridges["bridge_bc"].status, "complete")
        project_id, status = manager.create_project(site_id="site_c", structure_type="house")
        self.assertEqual(status, "created")
        self.assertTrue(manager.deliver_resource(project_id, "bricks", quantity=1))

    def test_site_a_zero_capacity_blocks_project_creation(self):
        manager = ConstructionManager(parameters={"site_a_capacity": 0, "site_b_capacity": 2, "site_c_capacity": 2})
        project_id, reason = manager.create_project("site_a", "house")
        self.assertIsNone(project_id)
        self.assertEqual(reason, "site_capacity_reached")

    def test_finite_resources_are_consumed(self):
        manager = ConstructionManager(parameters={"pile_a_quantity": 3, "pile_c_quantity": 0, "site_a_capacity": 1, "site_b_capacity": 3})
        project_id, status = manager.create_project("site_b", "house")
        self.assertEqual(status, "created")
        self.assertTrue(manager.deliver_resource(project_id, "bricks", quantity=1))
        self.assertEqual(manager.resource_nodes["pile_a"].quantity, 2)
        self.assertTrue(manager.deliver_resource(project_id, "bricks", quantity=1))
        self.assertTrue(manager.deliver_resource(project_id, "bricks", quantity=1))
        self.assertFalse(manager.deliver_resource(project_id, "bricks", quantity=1))

    def test_transport_timing_and_carry_capacity(self):
        manager = ConstructionManager(parameters={"move_time_per_unit": 4, "carry_capacity": 1})
        self.assertFalse(manager.reserve_transport("Architect", "site_a", "site_b", quantity=2))
        self.assertTrue(manager.reserve_transport("Architect", "site_a", "site_b", quantity=1))
        self.assertTrue(manager.is_agent_transporting("Architect"))
        for _ in range(3):
            manager.update()
            self.assertTrue(manager.is_agent_transporting("Architect"))
        manager.update()
        self.assertFalse(manager.is_agent_transporting("Architect"))

    def test_task_default_config_loads(self):
        cfg_path = Path("config/tasks/mars_colony/construction_parameters.json")
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["bridge_bc_cost"], 20)
        self.assertEqual(payload["carry_capacity"], 1)
        self.assertEqual(payload["site_a_capacity"], 0)

    def test_experiment_parameters_propagate_to_simulation(self):
        custom = {
            "pile_a_quantity": 11,
            "pile_c_quantity": 7,
            "housing_cost": 4,
            "greenhouse_cost": 5,
            "water_generator_cost": 6,
            "bridge_bc_cost": 9,
            "site_a_capacity": 2,
            "site_b_capacity": 3,
            "site_c_capacity": 4,
            "move_time_per_unit": 2,
            "carry_capacity": 1,
        }
        sim = SimulationState(agent_configs=[], num_runs=1, construction_parameters=custom, brain_backend="rule_brain")
        self.assertEqual(sim.construction_parameters["pile_a_quantity"], 11)
        self.assertEqual(sim.environment.construction.parameters["bridge_bc_cost"], 9)
        self.assertEqual(sim.environment.construction.sites["site_b"].capacity, 3)

    def test_any_structure_type_can_be_created_at_any_buildable_site(self):
        manager = ConstructionManager(parameters={"site_a_capacity": 0, "site_b_capacity": 4, "site_c_capacity": 4})
        self.assertTrue(manager.build_bridge_bc(quantity=20))
        for site_id in ("site_b", "site_c"):
            for structure_type in ("house", "greenhouse", "water_generator"):
                project_id, status = manager.create_project(site_id=site_id, structure_type=structure_type)
                self.assertEqual(status, "created")
                self.assertEqual(manager.projects[project_id]["type"], structure_type)


if __name__ == "__main__":
    unittest.main()
