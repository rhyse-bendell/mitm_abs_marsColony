import unittest

from modules.construction import ConstructionManager

try:
    from interface import MarsColonyInterface
except ModuleNotFoundError:  # pragma: no cover - env dependent
    MarsColonyInterface = None

try:
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Polygon, Rectangle
except ModuleNotFoundError:  # pragma: no cover - env dependent
    Figure = None
    Circle = Polygon = Rectangle = None


class ConstructionVisualizationTests(unittest.TestCase):
    def test_scene_data_includes_explicit_sites_and_structures(self):
        manager = ConstructionManager()
        scene = manager.get_construction_scene_data()

        self.assertIn("sites", scene)
        self.assertIn("structures", scene)
        self.assertIn("connectors", scene)
        self.assertGreaterEqual(len(scene["sites"]), 1)

        site = scene["sites"][0]
        self.assertIn("site_id", site)
        self.assertIn("position", site)
        self.assertIn("label", site)
        self.assertIn("project_ids", site)

        structure = scene["structures"][0]
        for field in (
            "project_id",
            "site_id",
            "structure_type",
            "progress",
            "status",
            "correct",
            "validated_complete",
            "resource_complete",
            "builders",
        ):
            self.assertIn(field, structure)

    def test_scene_data_tracks_project_states_without_logic_changes(self):
        manager = ConstructionManager()
        manager.deliver_resource("Build_Table_A", "bricks", quantity=12)
        manager.mark_validated("Build_Table_A", is_valid=True)
        manager.deliver_resource("Build_Table_B", "bricks", quantity=7)

        scene = manager.get_construction_scene_data()
        by_id = {row["project_id"]: row for row in scene["structures"]}
        self.assertEqual(by_id["Build_Table_A"]["status"], "complete")
        self.assertTrue(by_id["Build_Table_A"]["validated_complete"])
        self.assertGreater(by_id["Build_Table_B"]["progress"], 0.0)
        self.assertLess(by_id["Build_Table_B"]["progress"], 1.0)

    def test_semantic_visual_mapping_uses_symbol_types(self):
        if MarsColonyInterface is None:
            self.skipTest("interface module unavailable")

        self.assertEqual(
            MarsColonyInterface._map_structure_visual({"structure_type": "house"})["symbol"],
            "house",
        )
        self.assertEqual(
            MarsColonyInterface._map_structure_visual({"structure_type": "greenhouse"})["symbol"],
            "greenhouse",
        )
        self.assertEqual(
            MarsColonyInterface._map_structure_visual({"structure_type": "water_generator"})["symbol"],
            "water_generator",
        )

    def test_render_draws_neutral_site_containers_and_nested_structures(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface/matplotlib unavailable")

        scene = {
            "sites": [{"site_id": "site_a", "position": (4.0, 4.0), "label": "Site A"}],
            "structures": [
                {"project_id": "H", "site_id": "site_a", "structure_type": "house", "progress": 0.3, "status": "in_progress", "correct": True},
                {"project_id": "G", "site_id": "site_a", "structure_type": "greenhouse", "progress": 0.6, "status": "in_progress", "correct": True},
            ],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        site_circles = [p for p in ax.patches if isinstance(p, Circle) and abs(p.radius - 0.95) < 1e-9]
        self.assertEqual(len(site_circles), 1)

        # Structure centers should be inside the same site footprint, not detached anchor pairs.
        structure_rects = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_linewidth() >= 1.7]
        self.assertGreaterEqual(len(structure_rects), 2)
        sx, sy = site_circles[0].center
        for rect in structure_rects[:2]:
            cx = rect.get_x() + rect.get_width() / 2.0
            cy = rect.get_y() + rect.get_height() / 2.0
            self.assertLessEqual(((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5, 0.95)

    def test_symbolic_paths_are_distinct_for_house_greenhouse_water(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface/matplotlib unavailable")

        scene = {
            "sites": [
                {"site_id": "s1", "position": (3.5, 4.0), "label": "S1"},
                {"site_id": "s2", "position": (5.0, 4.0), "label": "S2"},
                {"site_id": "s3", "position": (6.5, 4.0), "label": "S3"},
            ],
            "structures": [
                {"project_id": "H", "site_id": "s1", "structure_type": "house", "progress": 0.5, "status": "in_progress", "correct": True},
                {"project_id": "G", "site_id": "s2", "structure_type": "greenhouse", "progress": 0.5, "status": "in_progress", "correct": True},
                {"project_id": "W", "site_id": "s3", "structure_type": "water_generator", "progress": 0.5, "status": "in_progress", "correct": True},
            ],
            "connectors": [],
        }

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        self.assertGreaterEqual(len([p for p in ax.patches if isinstance(p, Polygon)]), 1)  # house roof
        self.assertGreaterEqual(len([p for p in ax.patches if isinstance(p, Circle) and abs(p.radius - 0.30) < 1e-9]), 1)  # water core
        self.assertGreaterEqual(len(ax.lines), 3)  # greenhouse frame + water cue lines

    def test_progress_fill_is_geometry_based_not_alpha_only(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface/matplotlib unavailable")

        scene = {
            "sites": [{"site_id": "s1", "position": (4.0, 4.0), "label": "S1"}],
            "structures": [{"project_id": "G", "site_id": "s1", "structure_type": "greenhouse", "progress": 0.25, "status": "in_progress", "correct": True}],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        fill_rects = [
            p
            for p in ax.patches
            if isinstance(p, Rectangle)
            and abs(p.get_width() - 0.74) < 1e-9
            and p.get_facecolor()[3] > 0
        ]
        self.assertTrue(any(abs(rect.get_height() - (0.42 * 0.25)) < 1e-9 for rect in fill_rects))

    def test_state_overlays_and_secondary_labels(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface/matplotlib unavailable")

        scene = {
            "sites": [
                {"site_id": "v", "position": (4.0, 4.0), "label": "Site V"},
                {"site_id": "x", "position": (6.0, 4.0), "label": "Site X"},
            ],
            "structures": [
                {"project_id": "V", "site_id": "v", "structure_type": "house", "progress": 1.0, "resource_complete": True, "validated_complete": False, "correct": True},
                {"project_id": "X", "site_id": "x", "structure_type": "house", "progress": 0.5, "status": "needs_repair", "resource_complete": False, "validated_complete": False, "correct": False},
            ],
            "connectors": [],
        }

        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        texts = [t.get_text() for t in ax.texts]
        self.assertIn("awaiting validation", texts)
        self.assertIn("Site V", texts)
        self.assertNotIn("Build_Table_A", texts)
        self.assertGreaterEqual(len(ax.lines), 2)  # invalid X overlay


if __name__ == "__main__":
    unittest.main()
