import unittest
import math

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
    @staticmethod
    def _site_radius():
        return MarsColonyInterface._site_circle_style()["radius"]

    @staticmethod
    def _outline_pile_centers(ax):
        outlines = []
        for patch in ax.patches:
            if not isinstance(patch, Rectangle):
                continue
            if abs(patch.get_width() - MarsColonyInterface._PILE_DRAW_SIZE) > 1e-9:
                continue
            if abs(patch.get_height() - MarsColonyInterface._PILE_DRAW_SIZE) > 1e-9:
                continue
            if patch.get_linewidth() < 1.6:
                continue
            outlines.append((patch.get_x() + patch.get_width() / 2.0, patch.get_y() + patch.get_height() / 2.0))
        return outlines

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

    def test_rendered_resource_piles_are_outside_associated_site_circles(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface/matplotlib unavailable")

        manager = ConstructionManager()
        scene = manager.get_construction_scene_data()
        site_lookup = {s["site_id"]: tuple(s["position"]) for s in scene["sites"]}

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)
        pile_centers = self._outline_pile_centers(ax)
        self.assertEqual(len(pile_centers), 2)

        radius = self._site_radius()
        half_diag = MarsColonyInterface._PILE_DRAW_SIZE * math.sqrt(2.0) / 2.0
        for pile in scene["resource_piles"]:
            sx, sy = site_lookup[pile["site_id"]]
            px, py = min(pile_centers, key=lambda c: math.hypot(c[0] - sx, c[1] - sy))
            center_distance = math.hypot(px - sx, py - sy)
            self.assertGreaterEqual(center_distance - half_diag, radius)

    def test_rendered_bridge_ab_segment_is_trimmed_and_still_overlaps_site_edges(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface/matplotlib unavailable")

        manager = ConstructionManager()
        scene = manager.get_construction_scene_data()
        site_lookup = {s["site_id"]: tuple(s["position"]) for s in scene["sites"]}
        a = site_lookup["site_a"]
        b = site_lookup["site_b"]
        center_distance = math.hypot(b[0] - a[0], b[1] - a[1])
        radius = self._site_radius()

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)
        self.assertEqual(len(ax.lines), 1)
        line = ax.lines[0]
        xdata = list(line.get_xdata())
        ydata = list(line.get_ydata())
        self.assertEqual(len(xdata), 2)
        self.assertEqual(len(ydata), 2)
        start = (xdata[0], ydata[0])
        end = (xdata[1], ydata[1])
        rendered_length = math.hypot(end[0] - start[0], end[1] - start[1])
        self.assertLess(rendered_length, center_distance)

        start_inside = math.hypot(start[0] - a[0], start[1] - a[1])
        end_inside = math.hypot(end[0] - b[0], end[1] - b[1])
        self.assertLess(start_inside, radius)
        self.assertLess(end_inside, radius)
        self.assertGreater(start_inside, radius - 0.3)
        self.assertGreater(end_inside, radius - 0.3)


if __name__ == "__main__":
    unittest.main()
