import unittest

from modules.construction import ConstructionManager


class PilePickupAccountingTests(unittest.TestCase):
    @staticmethod
    def _project(cm):
        cm.sites["site_a"].capacity = 5
        project_id, _ = cm.create_project(site_id="site_a", structure_type="house", author="system")
        return project_id

    def test_pickup_removes_brick_from_pile_and_inventory(self):
        cm = ConstructionManager()
        before = cm.resource_nodes["pile_a"].quantity
        self.assertTrue(cm.take_from_pile("pile_a", 1))
        self.assertEqual(cm.resource_nodes["pile_a"].quantity, before - 1)
        self.assertEqual(cm.site_resource_inventory["site_a"], before - 1)

    def test_two_agents_cannot_take_the_same_last_brick(self):
        cm = ConstructionManager()
        cm.site_resource_inventory["site_a"] = 1
        cm.resource_nodes["pile_a"].quantity = 1
        self.assertTrue(cm.take_from_pile("pile_a", 1))
        self.assertFalse(cm.take_from_pile("pile_a", 1))
        self.assertEqual(cm.resource_nodes["pile_a"].quantity, 0)

    def test_failed_take_resyncs_stale_pile_quantity(self):
        cm = ConstructionManager()
        cm.site_resource_inventory["site_a"] = 0
        self.assertFalse(cm.take_from_pile("pile_a", 1))
        self.assertEqual(cm.resource_nodes["pile_a"].quantity, 0)

    def test_carried_delivery_does_not_consume_pile_again(self):
        cm = ConstructionManager()
        pid = self._project(cm)
        self.assertTrue(cm.take_from_pile("pile_a", 1))
        after_pickup = cm.site_resource_inventory["site_a"]
        self.assertTrue(cm.deliver_resource(pid, "bricks", quantity=1, from_carried=True))
        self.assertEqual(cm.site_resource_inventory["site_a"], after_pickup)

    def test_carried_delivery_succeeds_when_pile_is_empty(self):
        cm = ConstructionManager()
        pid = self._project(cm)
        cm.site_resource_inventory["site_a"] = 1
        cm.resource_nodes["pile_a"].quantity = 1
        self.assertTrue(cm.take_from_pile("pile_a", 1))
        self.assertFalse(cm.deliver_resource(pid, "bricks", quantity=1))
        self.assertTrue(cm.deliver_resource(pid, "bricks", quantity=1, from_carried=True))

    def test_return_to_pile_restores_brick(self):
        cm = ConstructionManager()
        before = cm.resource_nodes["pile_c"].quantity
        cm.take_from_pile("pile_c", 1)
        self.assertTrue(cm.return_to_pile("pile_c", 1))
        self.assertEqual(cm.resource_nodes["pile_c"].quantity, before)
        self.assertEqual(cm.site_resource_inventory["site_c"], before)


if __name__ == "__main__":
    unittest.main()
