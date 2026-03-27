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

    def test_agent_interaction_state_format_uses_runtime_snapshot(self):
        try:
            interface_mod = importlib.import_module("interface")
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")
            return

        line = interface_mod.MarsColonyInterface._format_agent_interaction_state(
            {
                "display_name": "Architect",
                "role": "Architect",
                "control_state": {"mode": "CONSTRUCT", "previous_mode": "LOGISTICS"},
                "method_state": {"active_method_id": "ConstructProject", "active_method_step": "start_or_continue_construction"},
                "planner_state": {"status": "in_flight"},
                "dik_integration_state": {"status": "running"},
                "inspect_session": {"state": "inspecting"},
                "transport_state": {"stage": "deliver"},
                "current_plan_id": "plan-1",
                "current_target": "Build_A",
            }
        )
        self.assertIn("mode=CONSTRUCT", line)
        self.assertIn("method=ConstructProject", line)
        self.assertIn("planner=in_flight", line)
        self.assertIn("movement=targeted", line)

    def test_agent_state_panel_format_contains_policy_snapshot(self):
        try:
            interface_mod = importlib.import_module("interface")
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")
            return

        rendered = interface_mod.MarsColonyInterface._format_agent_state_panel(
            {
                "display_name": "Engineer",
                "role": "Engineer",
                "control_state": {
                    "mode": "RECOVERY",
                    "previous_mode": "ACQUIRE_DIK",
                    "mode_dwell_steps": 2,
                    "last_transition_reason": "loop_pressure_bias_recovery",
                    "policy_snapshot": {"top_features": {"loop_pressure": 1.0}},
                },
                "method_state": {
                    "active_method_id": "RepairProject",
                    "active_method_step": "attempt_repair",
                    "step_retry_count": 1,
                    "source_cooldowns": {"Team_Info": 14},
                },
                "planner_state": {"status": "idle", "request_id": "req-1", "last_result_request_id": "req-0"},
                "dik_integration_state": {"status": "idle"},
                "fallback_bootstrap": {"active": False, "stage": "complete"},
                "inspect_session": {"state": "idle"},
                "inspect_pursuit": {"no_progress_ticks": 1, "blocked_attempts": 2},
                "transport_state": {"stage": "idle", "bound_project_id": None},
                "top_goals": [{"goal_id": "repair_bridge"}],
                "current_plan_id": "plan-2",
                "current_plan_method": "rule_brain_hierarchical_policy_v1",
                "next_action": "reassess_plan",
                "current_target": None,
                "last_status": "Recovering from assistance loop",
            }
        )
        self.assertIn("Macro mode: RECOVERY", rendered)
        self.assertIn("Method: RepairProject", rendered)
        self.assertIn("Policy features", rendered)


if __name__ == "__main__":
    unittest.main()
