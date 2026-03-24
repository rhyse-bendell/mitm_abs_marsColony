import unittest

from modules.construction import ConstructionManager

try:
    from interface import MarsColonyInterface
except ModuleNotFoundError:  # pragma: no cover - env dependent
    MarsColonyInterface = None

try:
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle
except ModuleNotFoundError:  # pragma: no cover - env dependent
    Figure = None
    Circle = Rectangle = None


class ConstructionVisualizationTests(unittest.TestCase):
    def test_mission_start_scene_layers_and_counts(self):
        manager = ConstructionManager()
        scene = manager.get_construction_scene_data()

        self.assertEqual(len(scene["sites"]), 3)
        self.assertEqual(len(scene["resource_piles"]), 2)
        self.assertEqual(len(scene["bridges"]), 1)
        self.assertEqual(scene["bridges"][0]["bridge_id"], "bridge_ab")
        self.assertEqual(scene["structures"], [])

    def test_started_structures_only_appear_after_start(self):
        manager = ConstructionManager()
        self.assertEqual(manager.get_construction_scene_data()["structures"], [])
        manager.assign_builder("Build_Table_A", "Architect")
        scene = manager.get_construction_scene_data()
        self.assertEqual(len(scene["structures"]), 1)
        self.assertEqual(scene["structures"][0]["site_id"], "site_a")

    def test_structure_progress_and_color_mapping(self):
        manager = ConstructionManager()
        manager.assign_builder("Build_Table_A", "Architect")
        manager.assign_builder("Build_Table_B", "Engineer")
        manager.build_bridge_bc(quantity=20)
        manager.assign_builder("Build_Table_C", "Botanist")
        manager.deliver_resource("Build_Table_B", "bricks", quantity=5)

        scene = manager.get_construction_scene_data()
        by_id = {row["project_id"]: row for row in scene["structures"]}
        self.assertEqual(by_id["Build_Table_A"]["color"], "#c6362f")
        self.assertEqual(by_id["Build_Table_B"]["color"], "#2f8f46")
        self.assertEqual(by_id["Build_Table_C"]["color"], "#2f6fbf")
        self.assertGreater(by_id["Build_Table_B"]["progress"], 0.0)
        self.assertLess(by_id["Build_Table_B"]["progress"], 1.0)

    def test_resource_pile_fill_reflects_remaining_quantity(self):
        manager = ConstructionManager(parameters={"pile_a_quantity": 100, "pile_c_quantity": 100})
        manager.deliver_resource("Build_Table_A", "bricks", quantity=50)
        scene = manager.get_construction_scene_data()
        piles = {p["pile_id"]: p for p in scene["resource_piles"]}
        self.assertEqual(piles["pile_a"]["remaining"], 50)
        self.assertAlmostEqual(piles["pile_a"]["fill_fraction"], 0.5)

    def test_render_draw_order_and_literals(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface/matplotlib unavailable")

        manager = ConstructionManager()
        scene = manager.get_construction_scene_data()
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        site_circles = [p for p in ax.patches if isinstance(p, Circle) and abs(p.radius - 0.86) < 1e-9]
        self.assertEqual(len(site_circles), 3)
        pile_squares = [p for p in ax.patches if isinstance(p, Rectangle) and abs(p.get_width() - 0.24) < 1e-9]
        self.assertEqual(len(pile_squares), 4)  # 2 outlines + 2 fills
        self.assertEqual(len(ax.lines), 1)  # AB bridge only at mission start

        manager.assign_builder("Build_Table_B", "Engineer")
        scene_started = manager.get_construction_scene_data()
        fig2 = Figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax2, scene_started)
        structure_squares = [p for p in ax2.patches if isinstance(p, Rectangle) and abs(p.get_width() - 0.28) < 1e-9]
        self.assertGreaterEqual(len(structure_squares), 1)


if __name__ == "__main__":
    unittest.main()
