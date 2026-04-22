import unittest

from modules.environment import Environment
from modules.simulation import SimulationState
from modules.task_model import load_task_model


class MarsConstructionDynamicModelTests(unittest.TestCase):
    def test_no_startup_project_seeding(self):
        sim = SimulationState(agent_configs=[], num_runs=1, brain_backend="rule_brain")
        try:
            self.assertEqual(sim.environment.construction.projects, {})
        finally:
            sim.stop()

    def test_site_a_capacity_zero_preserved(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        self.assertEqual(env.construction.sites["site_a"].capacity, 0)
        project_id, reason = env.construction.create_project("site_a", "house")
        self.assertIsNone(project_id)
        self.assertEqual(reason, "site_capacity_reached")

    def test_dynamic_project_creation_up_to_capacity(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model, construction_parameters={"site_b_capacity": 2})
        p1, s1 = env.construction.create_project("site_b", "house")
        p2, s2 = env.construction.create_project("site_b", "greenhouse")
        p3, s3 = env.construction.create_project("site_b", "water_generator")
        self.assertEqual((s1, s2), ("created", "created"))
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertIsNone(p3)
        self.assertEqual(s3, "site_capacity_reached")

    def test_legacy_target_alias_is_not_live_interaction_target(self):
        model = load_task_model("mars_colony")
        env = Environment(task_model=model)
        self.assertIn("Build_Site_A", env.interaction_targets)
        self.assertNotIn("Build_Table_A", env.interaction_targets)
        self.assertEqual(env._canonical_build_target("Build_Table_A"), "Build_Site_A")


if __name__ == "__main__":
    unittest.main()
