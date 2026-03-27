import importlib
import tkinter as tk
import unittest


class _FakeLogger:
    def __init__(self, rows):
        self._rows = list(rows)

    def get_recent_interactions(self, count=180):
        return self._rows[-count:]


class _FakeSim:
    def __init__(self, rows, time_value=100.0):
        self.logger = _FakeLogger(rows)
        self.time = float(time_value)


class TestInteractionTabResilience(unittest.TestCase):
    def _build_app(self):
        try:
            interface_mod = importlib.import_module("interface")
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")
            return None
        MarsColonyInterface = interface_mod.MarsColonyInterface
        try:
            app = MarsColonyInterface()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable in test environment: {exc}")
            return None
        app.root.withdraw()
        return app

    def test_update_interaction_tab_reports_empty_logger_rows(self):
        app = self._build_app()
        if app is None:
            return
        try:
            app.sim = _FakeSim(rows=[], time_value=25.0)
            app.update_interaction_tab()
            body = app.interaction_list.get("1.0", "end-1c")
            self.assertIn("No interaction events are being logged yet", body)
            self.assertIn("total=0", app.interaction_status_var.get())

            text_items = [app.interaction_canvas.itemcget(item, "text") for item in app.interaction_canvas.find_all()]
            self.assertTrue(any("No interaction events are being logged yet" in text for text in text_items))
        finally:
            app.root.destroy()

    def test_update_interaction_tab_reports_filter_empty_rows(self):
        app = self._build_app()
        if app is None:
            return
        try:
            rows = [
                {"time": 90.0, "interaction_type": "handoff", "agent_id": "Architect", "source_node": "a", "target_node": "b"},
                {"time": 91.0, "interaction_type": "assist", "agent_id": "Engineer", "source_node": "b", "target_node": "c"},
            ]
            app.sim = _FakeSim(rows=rows, time_value=100.0)
            app.interaction_type_filter.set("inspect")
            app.update_interaction_tab()
            body = app.interaction_list.get("1.0", "end-1c")
            self.assertIn("No interaction events match the current filters/window", body)
            self.assertIn("total=2", app.interaction_status_var.get())
            self.assertIn("shown=0", app.interaction_status_var.get())
        finally:
            app.root.destroy()

    def test_interaction_row_normalization_handles_missing_optional_keys(self):
        try:
            interface_mod = importlib.import_module("interface")
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")
            return
        cls = interface_mod.MarsColonyInterface
        normalized = cls._normalize_interaction_row({"time": "12.5", "agent_id": "Architect"}, index=1)
        self.assertEqual(12.5, normalized["time"])
        self.assertEqual("unknown", normalized["interaction_type"])
        self.assertEqual("Architect", normalized["source_node"])
        self.assertEqual("-", normalized["target_node"])
        self.assertEqual("unknown", normalized["status"])

