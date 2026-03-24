import json
import unittest
from pathlib import Path

from modules.construction import ConstructionManager
from modules.simulation import SimulationState


class ConstructionSystemRuleTests(unittest.TestCase):
    def test_site_c_blocked_until_bridge_complete(self):
        manager = ConstructionManager()
        ok = manager.deliver_resource("Build_Table_C", "bricks", quantity=1)
        self.assertFalse(ok)
        self.assertFalse(manager.projects["Build_Table_C"]["started"])

    def test_bridge_build_enables_site_c(self):
        manager = ConstructionManager()
        self.assertEqual(manager.bridges["bridge_bc"].status, "not_started")
        self.assertFalse(manager.build_bridge_bc(quantity=10))
        self.assertEqual(manager.bridges["bridge_bc"].status, "in_progress")
        self.assertTrue(manager.build_bridge_bc(quantity=10))
        self.assertEqual(manager.bridges["bridge_bc"].status, "complete")
        self.assertTrue(manager.deliver_resource("Build_Table_C", "bricks", quantity=1))

    def test_site_capacity_enforced(self):
        manager = ConstructionManager(parameters={"site_a_capacity": 1, "site_b_capacity": 8, "site_c_capacity": 16})
        manager.start_project("Build_Table_A")
        manager.projects["Build_Table_A"]["id"] = "Build_Table_A_1"
        # Force another project into same site for direct capacity check.
        manager.projects["Extra_A"] = dict(manager.projects["Build_Table_A"])
        manager.projects["Extra_A"]["id"] = "Extra_A"
        manager.projects["Extra_A"]["site_id"] = "site_a"
        manager.projects["Extra_A"]["started"] = False
        ok, reason = manager.start_project("Extra_A")
        self.assertFalse(ok)
        self.assertEqual(reason, "site_capacity_reached")

    def test_finite_resources_are_consumed(self):
        manager = ConstructionManager(parameters={"pile_a_quantity": 3, "pile_c_quantity": 0})
        self.assertTrue(manager.deliver_resource("Build_Table_A", "bricks", quantity=1))
        self.assertEqual(manager.resource_nodes["pile_a"].quantity, 2)
        self.assertTrue(manager.deliver_resource("Build_Table_A", "bricks", quantity=1))
        self.assertTrue(manager.deliver_resource("Build_Table_A", "bricks", quantity=1))
        self.assertFalse(manager.deliver_resource("Build_Table_A", "bricks", quantity=1))

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


if __name__ == "__main__":
    unittest.main()
