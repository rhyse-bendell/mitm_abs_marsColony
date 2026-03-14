import importlib
import unittest
import tkinter as tk


class TestInterfaceDashboardStructure(unittest.TestCase):
    def test_dashboard_tab_and_widgets_exist(self):
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

        try:
            app.root.withdraw()
            tab_texts = [app.notebook.tab(tab_id, "text") for tab_id in app.notebook.tabs()]
            self.assertIn("Dashboard", tab_texts)
            self.assertIn("Environment", tab_texts)
            self.assertIn("Event Monitor", tab_texts)

            self.assertTrue(hasattr(app, "dashboard_canvas"))
            self.assertTrue(hasattr(app, "dashboard_agent_activity_text"))
            self.assertTrue(hasattr(app, "dashboard_interaction_state_text"))
            self.assertTrue(hasattr(app, "dashboard_zone_state_text"))
            self.assertTrue(hasattr(app, "dashboard_agent_state"))
            self.assertTrue(hasattr(app, "dashboard_construction_text"))
            self.assertTrue(hasattr(app, "system_log_text"))
        finally:
            app.root.destroy()


if __name__ == "__main__":
    unittest.main()
