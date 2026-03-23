import unittest
import math

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
        sites = scene.get("sites", [])

        self.assertIn("Build_Table_A", by_id)
        self.assertIn("Build_Table_B", by_id)
        self.assertTrue(all(row.get("site_id") for row in by_id.values()))
        self.assertGreaterEqual(len(sites), 1)
        self.assertTrue(by_id["Build_Table_A"]["validated_complete"])
        self.assertEqual(by_id["Build_Table_A"]["status"], "complete")
        self.assertGreater(by_id["Build_Table_B"]["progress"], 0.0)
        self.assertLess(by_id["Build_Table_B"]["progress"], 1.0)

    def test_scene_data_includes_explicit_sites_layer(self):
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

    def test_site_local_offsets_support_single_pair_and_ring(self):
        if MarsColonyInterface is None:
            self.skipTest("interface module unavailable")
        self.assertEqual(MarsColonyInterface._site_local_offsets(1), [(0.0, 0.0)])
        self.assertEqual(
            MarsColonyInterface._site_local_offsets(2),
            [(-0.44, 0.0), (0.44, 0.0)],
        )
        offsets = MarsColonyInterface._site_local_offsets(3)
        self.assertEqual(len(offsets), 3)
        radii = [math.hypot(x, y) for x, y in offsets]
        for radius in radii:
            self.assertAlmostEqual(radius, 0.44, places=6)

    def test_structure_visual_mapping_prefers_semantic_tokens(self):
        if MarsColonyInterface is None:
            self.skipTest("interface module unavailable")
        self.assertEqual(
            MarsColonyInterface._map_structure_visual({"structure_type": "housing"}),
            {"shape": "square", "color": "red"},
        )
        self.assertEqual(
            MarsColonyInterface._map_structure_visual({"name": "Food Dome Project"}),
            {"shape": "rectangle", "color": "green"},
        )
        self.assertEqual(
            MarsColonyInterface._map_structure_visual({"project_id": "Water_Module_A"}),
            {"shape": "circle", "color": "blue"},
        )

    def test_progress_fraction_and_fill_geometry(self):
        if MarsColonyInterface is None:
            self.skipTest("interface module unavailable")
        self.assertEqual(MarsColonyInterface._progress_fill_fraction({"progress": None}), 0.0)
        self.assertEqual(MarsColonyInterface._progress_fill_fraction({"progress": 1.7}), 1.0)
        self.assertAlmostEqual(MarsColonyInterface._progress_fill_fraction({"progress": 0.25}), 0.25)

        left, bottom, width, fill_h = MarsColonyInterface._rect_fill_geometry(5.0, 4.0, 1.0, 2.0, 0.25)
        self.assertEqual((left, bottom, width), (4.5, 3.0, 1.0))
        self.assertAlmostEqual(fill_h, 0.5)

        clip_left, clip_bottom, clip_width, clip_height = MarsColonyInterface._circle_fill_clip_geometry(6.0, 3.0, 0.4, 0.5)
        self.assertAlmostEqual(clip_left, 5.6)
        self.assertAlmostEqual(clip_bottom, 2.6)
        self.assertAlmostEqual(clip_width, 0.8)
        self.assertAlmostEqual(clip_height, 0.4)

    def test_overlay_state_selection(self):
        if MarsColonyInterface is None:
            self.skipTest("interface module unavailable")
        self.assertEqual(
            MarsColonyInterface._project_overlay_state({"resource_complete": True, "validated_complete": False, "correct": True}),
            "ready_for_validation",
        )
        self.assertEqual(
            MarsColonyInterface._project_overlay_state({"resource_complete": True, "validated_complete": True, "correct": True}),
            "validated",
        )
        self.assertEqual(
            MarsColonyInterface._project_overlay_state({"status": "needs_repair", "correct": False}),
            "invalid",
        )

    def test_render_uses_neutral_site_container_and_offsets_for_shared_site(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface module unavailable")
        scene = {
            "sites": [
                {"site_id": "site_a", "position": (4.0, 4.0), "label": "Site A"},
            ],
            "structures": [
                {"project_id": "A", "site_id": "site_a", "name": "House A", "structure_type": "house", "position": (1.0, 1.0), "progress": 0.4, "status": "in_progress", "correct": True},
                {"project_id": "B", "site_id": "site_a", "name": "Greenhouse B", "structure_type": "greenhouse", "position": (1.0, 1.0), "progress": 0.7, "status": "in_progress", "correct": True},
            ],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        site_containers = [
            p for p in ax.patches
            if hasattr(p, "radius") and abs(p.radius - 0.90) < 1e-9
        ]
        self.assertEqual(len(site_containers), 1)
        self.assertNotEqual(tuple(site_containers[0].get_facecolor()[:3]), (0.0, 0.0, 0.0))

        house_outline, greenhouse_outline = [
            p for p in ax.patches
            if p.__class__.__name__ == "Rectangle" and abs(p.get_linewidth() - 2.0) < 1e-9
        ]
        house_center_x = house_outline.get_x() + house_outline.get_width() / 2.0
        greenhouse_center_x = greenhouse_outline.get_x() + greenhouse_outline.get_width() / 2.0
        self.assertNotAlmostEqual(house_center_x, greenhouse_center_x, places=3)

    def test_single_site_layout_centers_structure_inside_site(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface module unavailable")
        scene = {
            "sites": [
                {"site_id": "site_h", "position": (5.0, 4.0), "label": "Site H"},
            ],
            "structures": [
                {"project_id": "H", "site_id": "site_h", "name": "House H", "structure_type": "house", "position": (1.0, 1.0), "progress": 0.5, "status": "in_progress", "correct": True},
            ],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        site_container = next(
            p for p in ax.patches
            if hasattr(p, "radius") and abs(p.radius - 0.90) < 1e-9
        )
        house_outline = next(
            p for p in ax.patches
            if p.__class__.__name__ == "Rectangle"
            and abs(p.get_width() - 0.68) < 1e-9
            and abs(p.get_height() - 0.68) < 1e-9
            and tuple(p.get_facecolor()[:3]) == (0.0, 0.0, 0.0)
            and abs(p.get_linewidth() - 2.0) < 1e-9
        )
        structure_center = (
            house_outline.get_x() + house_outline.get_width() / 2.0,
            house_outline.get_y() + house_outline.get_height() / 2.0,
        )
        self.assertAlmostEqual(site_container.center[0], structure_center[0], places=6)
        self.assertAlmostEqual(site_container.center[1], structure_center[1], places=6)

    def test_progress_fill_applies_to_structure_glyph(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface module unavailable")
        scene = {
            "sites": [
                {"site_id": "site_g", "position": (5.0, 4.0), "label": "Site G"},
            ],
            "structures": [
                {"project_id": "G", "site_id": "site_g", "name": "Greenhouse G", "structure_type": "greenhouse", "position": (1.0, 1.0), "progress": 0.25, "status": "in_progress", "correct": True},
            ],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)
        fill_rects = [
            p for p in ax.patches
            if p.__class__.__name__ == "Rectangle"
            and tuple(p.get_facecolor()[:3]) == (0.0, 0.5019607843137255, 0.0)
            and abs(p.get_width() - 0.94) < 1e-9
        ]
        self.assertTrue(any(abs(rect.get_height() - (0.58 * 0.25)) < 1e-9 for rect in fill_rects))

    def test_validation_and_invalid_overlays_render_when_present(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface module unavailable")
        scene = {
            "sites": [
                {"site_id": "site_v", "position": (4.0, 4.0), "label": "Site V"},
                {"site_id": "site_x", "position": (6.0, 4.0), "label": "Site X"},
            ],
            "structures": [
                {
                    "project_id": "V",
                    "site_id": "site_v",
                    "name": "House V",
                    "structure_type": "house",
                    "position": (4.0, 4.0),
                    "progress": 1.0,
                    "resource_complete": True,
                    "validated_complete": False,
                    "correct": True,
                },
                {
                    "project_id": "X",
                    "site_id": "site_x",
                    "name": "House X",
                    "structure_type": "house",
                    "position": (6.0, 4.0),
                    "progress": 0.6,
                    "resource_complete": False,
                    "validated_complete": False,
                    "correct": False,
                    "status": "needs_repair",
                },
            ],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        overlay_texts = [t.get_text() for t in ax.texts]
        self.assertIn("awaiting validation", overlay_texts)
        self.assertGreaterEqual(len(ax.lines), 2)

    def test_site_and_structure_labels_are_not_duplicated(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface module unavailable")
        scene = {
            "sites": [
                {"site_id": "site_a", "position": (4.0, 4.0), "label": "Site A"},
            ],
            "structures": [
                {"project_id": "Build_Table_A", "site_id": "site_a", "name": "House A", "structure_type": "house", "position": (4.0, 4.0), "progress": 0.4, "status": "in_progress", "correct": True},
                {"project_id": "Build_Table_B", "site_id": "site_a", "name": "Greenhouse B", "structure_type": "greenhouse", "position": (4.0, 4.0), "progress": 0.7, "status": "in_progress", "correct": True},
            ],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)

        texts = [t.get_text() for t in ax.texts]
        self.assertIn("Site A", texts)
        self.assertNotIn("Build_Table_A", texts)
        self.assertNotIn("Build_Table_B", texts)

    def test_structure_line_progress_renders_when_line_structure_present(self):
        if MarsColonyInterface is None or Figure is None:
            self.skipTest("interface module unavailable")
        scene = {
            "sites": [
                {"site_id": "site_pipe", "position": (4.0, 4.0), "label": "Site Pipe"},
            ],
            "structures": [
                {
                    "project_id": "Pipe_A",
                    "site_id": "site_pipe",
                    "name": "resource_link_pipe",
                    "structure_type": "connector",
                    "position": (4.0, 4.0),
                    "progress": 0.5,
                    "status": "in_progress",
                    "correct": True,
                }
            ],
            "connectors": [],
        }
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        MarsColonyInterface._draw_construction_scene(ax, scene)
        self.assertGreaterEqual(len(ax.lines), 2)

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
