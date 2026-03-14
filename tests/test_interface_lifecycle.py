import importlib
import tkinter as tk
import unittest


class _FakeSim:
    def __init__(self):
        self.time = 0.0
        self.update_calls = 0
        self.stop_calls = 0

    def update(self, dt):
        self.update_calls += 1
        self.time += dt

    def stop(self):
        self.stop_calls += 1


class TestInterfaceLifecycle(unittest.TestCase):
    def setUp(self):
        try:
            interface_mod = importlib.import_module("interface")
            self.MarsColonyInterface = interface_mod.MarsColonyInterface
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependency unavailable in test environment: {exc}")
            return

        try:
            self.app = self.MarsColonyInterface()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable in test environment: {exc}")
            return

        self.app.root.withdraw()

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.root.destroy()

    def test_run_loop_state_transitions_and_stop_flush(self):
        fake = _FakeSim()
        self.app.sim = fake
        self.app.run_state = self.app.STATE_IDLE

        scheduled = []
        self.app._schedule_next_tick = lambda: scheduled.append("scheduled")

        self.app.start_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_RUNNING)
        self.assertEqual(scheduled, ["scheduled"])
        self.assertEqual(str(self.app.start_button["state"]), "disabled")
        self.assertEqual(str(self.app.pause_button["state"]), "normal")
        self.assertEqual(str(self.app.stop_button["state"]), "normal")

        self.app.pause_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_PAUSED)
        self.assertEqual(str(self.app.start_button["state"]), "normal")
        self.assertEqual(str(self.app.pause_button["state"]), "disabled")
        self.assertEqual(str(self.app.stop_button["state"]), "normal")

        paused_time = fake.time
        self.app._run_loop_tick()
        self.assertEqual(fake.time, paused_time)

        self.app.start_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_RUNNING)

        self.app.stop_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_STOPPED)
        self.assertEqual(fake.stop_calls, 1)
        self.assertEqual(str(self.app.start_button["state"]), "normal")
        self.assertEqual(str(self.app.pause_button["state"]), "disabled")
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")

    def test_run_loop_tick_advances_sim_without_tab_switch(self):
        fake = _FakeSim()
        self.app.sim = fake
        self.app.run_state = self.app.STATE_RUNNING

        updates = {"environment": 0, "agents": 0, "events": 0, "dashboard": 0, "construction": 0}
        self.app.update_environment_plot = lambda frame=None: updates.__setitem__("environment", updates["environment"] + 1)
        self.app.update_agent_table = lambda: updates.__setitem__("agents", updates["agents"] + 1)
        self.app.update_event_monitor = lambda: updates.__setitem__("events", updates["events"] + 1)
        self.app.update_dashboard = lambda: updates.__setitem__("dashboard", updates["dashboard"] + 1)
        self.app._sync_construction_summaries = lambda: updates.__setitem__("construction", updates["construction"] + 1)

        scheduled = []
        self.app._schedule_next_tick = lambda: scheduled.append("scheduled")

        self.app._run_loop_tick()

        self.assertEqual(fake.update_calls, 1)
        self.assertAlmostEqual(fake.time, self.app.base_dt)
        self.assertEqual(updates, {"environment": 1, "agents": 1, "events": 1, "dashboard": 1, "construction": 1})
        self.assertEqual(scheduled, ["scheduled"])


if __name__ == "__main__":
    unittest.main()
