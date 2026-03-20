import unittest

from modules.construction import ConstructionManager

try:
    from interface import MarsColonyInterface
except ModuleNotFoundError:  # pragma: no cover - env dependent
    MarsColonyInterface = None

try:
    from matplotlib.figure import Figure
except ModuleNotFoundError:  # pragma: no cover - env dependent
    Figure = None


class ConstructionVisualizationTests(unittest.TestCase):
    def test_scene_data_includes_completed_and_in_progress_projects(self):
        manager = ConstructionManager()
        manager.deliver_resource("Build_Table_A", "bricks", quantity=12)
        manager.mark_validated("Build_Table_A", is_valid=True)
        manager.deliver_resource("Build_Table_B", "bricks", quantity=7)

        scene = manager.get_construction_scene_data()
        by_id = {row["project_id"]: row for row in scene["structures"]}

        self.assertIn("Build_Table_A", by_id)
        self.assertIn("Build_Table_B", by_id)
        self.assertTrue(by_id["Build_Table_A"]["validated_complete"])
        self.assertEqual(by_id["Build_Table_A"]["status"], "complete")
        self.assertGreater(by_id["Build_Table_B"]["progress"], 0.0)
        self.assertLess(by_id["Build_Table_B"]["progress"], 1.0)

    def test_scene_data_contains_type_specific_shape_and_color(self):
        manager = ConstructionManager()
        scene = manager.get_construction_scene_data()
        by_type = {row["structure_type"]: row for row in scene["structures"]}

        self.assertEqual(by_type["house"]["shape"], "square")
        self.assertEqual(by_type["house"]["color"], "red")
        self.assertEqual(by_type["greenhouse"]["shape"], "rectangle")
        self.assertEqual(by_type["greenhouse"]["color"], "green")
        self.assertEqual(by_type["water_generator"]["shape"], "circle")
        self.assertEqual(by_type["water_generator"]["color"], "blue")

    def test_scene_data_does_not_fabricate_connectors(self):
        manager = ConstructionManager()
        scene = manager.get_construction_scene_data()
        self.assertEqual(scene.get("connectors"), [])

    def test_construction_tab_render_path_draws_visual_elements(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface module unavailable")
        manager = ConstructionManager()
        manager.deliver_resource("Build_Table_C", "bricks", quantity=10)
        scene = manager.get_construction_scene_data()
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)

        MarsColonyInterface._draw_construction_scene(ax, scene)

        self.assertGreater(len(ax.patches), 0)
        self.assertGreater(len(ax.texts), 0)


if __name__ == "__main__":
    unittest.main()
