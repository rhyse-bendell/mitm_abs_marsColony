import importlib
import tkinter as tk
import unittest


class _SnapshotAgent:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def get_runtime_state_snapshot(self):
        return dict(self._snapshot)


class _FakeSim:
    def __init__(self, snapshots):
        self.agents = [_SnapshotAgent(snapshot) for snapshot in snapshots]


class TestAgentStateMachineHelpers(unittest.TestCase):
    def setUp(self):
        try:
            self.interface_mod = importlib.import_module("interface")
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")
            return
        self.cls = self.interface_mod.MarsColonyInterface

    def test_state_machine_builder_handles_partial_snapshot(self):
        graph = self.cls._state_machine_nodes_for_snapshot({"display_name": "Architect"})
        self.assertIn("BOOTSTRAP", graph["modes"])
        self.assertEqual(graph["current_mode"], "BOOTSTRAP")
        self.assertIsNone(graph["current_method"])
        self.assertEqual(graph["steps"], [])
        self.assertIn("support_badges", graph)

    def test_known_modes_methods_and_steps_are_deterministic(self):
        snapshot = {
            "control_state": {"mode": "LOGISTICS", "previous_mode": "COORDINATE", "mode_history": ["BOOTSTRAP", "ACQUIRE_DIK", "LOGISTICS"]},
            "method_state": {
                "active_method_id": "TransportResourcesToProject",
                "active_method_step": "pickup",
                "step_retry_count": 2,
                "source_cooldowns": {"Team_Info": 12},
                "source_exhaustion": {"Team_Info": {"exhausted": True}},
            },
            "planner_state": {"status": "degraded"},
            "inspect_pursuit": {"no_progress_ticks": 3, "blocked_attempts": 1},
        }
        graph = self.cls._state_machine_nodes_for_snapshot(snapshot)
        self.assertEqual(graph["current_mode"], "LOGISTICS")
        self.assertIn("TransportResourcesToProject", graph["methods"])
        self.assertIn("pickup", graph["steps"])
        self.assertIn("planner:degraded", graph["warnings"])
        self.assertIn("exhausted:Team_Info", graph["warnings"])

    def test_compute_state_layer_layout_reflows_with_viewport_width(self):
        nodes = [{"key": f"m{i}", "label": f"Long Method Label {i} Needs Wrapping"} for i in range(6)]
        narrow = self.cls._compute_state_layer_layout(nodes, 420, min_node_width=120, max_node_width=180)
        wide = self.cls._compute_state_layer_layout(nodes, 1280, min_node_width=120, max_node_width=220)
        self.assertLess(narrow["columns"], wide["columns"])
        self.assertGreater(narrow["content_height"], wide["content_height"])
        self.assertIn("\n", narrow["entries"][0]["label"])

    def test_compute_canvas_scrollregion_expands_to_content_bounds(self):
        region = self.cls._compute_canvas_scrollregion(400, 280, (0, 0, 920, 640))
        self.assertEqual(region, (0, 0, 930, 650))
        fallback = self.cls._compute_canvas_scrollregion(400, 280, None)
        self.assertEqual(fallback, (0, 0, 400, 280))


class TestUpdateAgentTableStateMachine(unittest.TestCase):
    def test_update_agent_table_renders_canvas_and_text(self):
        try:
            interface_mod = importlib.import_module("interface")
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")
            return
        MarsColonyInterface = interface_mod.MarsColonyInterface

        try:
            app = MarsColonyInterface()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable in test environment: {exc}")
            return

        snapshots = [
            {
                "agent_id": "architect",
                "display_name": "Architect",
                "role": "Architect",
                "control_state": {"mode": "CONSTRUCT", "previous_mode": "LOGISTICS", "mode_history": ["BOOTSTRAP", "LOGISTICS", "CONSTRUCT"]},
                "method_state": {"active_method_id": "ConstructProject", "active_method_step": "ensure_build_ready"},
                "planner_state": {"status": "in_flight"},
            },
            {
                "agent_id": "engineer",
                "display_name": "Engineer",
                "role": "Engineer",
                "control_state": {"mode": "ACQUIRE_DIK", "previous_mode": "BOOTSTRAP"},
                "method_state": {"active_method_id": "AcquireRoleSpecificGrounding", "active_method_step": "inspect_role_source", "step_retry_count": 1},
            },
        ]

        try:
            app.root.withdraw()
            app.sim = _FakeSim(snapshots)
            app.update_agent_table()

            self.assertEqual(len(app.agent_state_panels), 2)
            arch = app.agent_state_panels["architect"]
            self.assertIsInstance(arch["canvas"], tk.Canvas)
            self.assertTrue(arch["canvas"].bbox("all"))
            self.assertIn("viewport", arch)
            panel_text = arch["body"].get("1.0", "end-1c")
            self.assertIn("Macro mode: CONSTRUCT", panel_text)
        finally:
            app.root.destroy()


if __name__ == "__main__":
    unittest.main()
