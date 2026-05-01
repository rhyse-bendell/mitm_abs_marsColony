import unittest

from modules.construction import ConstructionManager


class TestCapacityBridgeUnlock(unittest.TestCase):
    def test_capacity_summary_and_lock_reason(self):
        cm = ConstructionManager()
        for i in range(cm.sites["site_b"].capacity):
            pid, reason = cm.create_project("site_b", structure_type="house", project_id_override=f"site_b_house_{i+1:03d}")
            self.assertIsNotNone(pid)
            self.assertEqual(reason, "created")
        pid, reason = cm.create_project("site_b", structure_type="house")
        self.assertIsNone(pid)
        self.assertEqual(reason, "site_capacity_reached")
        summary = {row["site_id"]: row for row in cm.get_site_capacity_summary()}
        self.assertEqual(summary["site_b"]["remaining"], 0)
        self.assertEqual(summary["site_c"]["remaining"], 16)
        self.assertFalse(summary["site_c"]["buildable"])
        self.assertEqual(summary["site_c"]["unlock_bridge_id"], "bridge_bc")

    def test_unlock_objective_identified(self):
        cm = ConstructionManager()
        for i in range(cm.sites["site_b"].capacity):
            cm.create_project("site_b", structure_type="house", project_id_override=f"site_b_house_{i+1:03d}")
        unlock = cm.get_next_capacity_unlock()
        self.assertIsNotNone(unlock)
        self.assertEqual(unlock["bridge_id"], "bridge_bc")

    def test_bridge_completion_unlocks_site_c(self):
        cm = ConstructionManager()
        self.assertFalse(cm._is_site_buildable("site_c"))
        self.assertEqual(cm.create_project("site_c", structure_type="house")[1], "site_locked_by_bridge")
        for _ in range(cm.bridges["bridge_bc"].required_resources):
            self.assertTrue(cm.build_bridge_bc(1)) if cm.bridges["bridge_bc"].delivered_resources + 1 >= cm.bridges["bridge_bc"].required_resources else self.assertFalse(cm.build_bridge_bc(1))
        self.assertEqual(cm.bridges["bridge_bc"].status, "complete")
        self.assertTrue(cm._is_site_buildable("site_c"))
        pid, reason = cm.create_project("site_c", structure_type="house")
        self.assertIsNotNone(pid)
        self.assertEqual(reason, "created")


if __name__ == "__main__":
    unittest.main()
