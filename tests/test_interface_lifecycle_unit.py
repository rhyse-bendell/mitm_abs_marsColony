import unittest
from unittest.mock import patch

try:
    from interface import MarsColonyInterface
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    MarsColonyInterface = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _FakeButton:
    def __init__(self):
        self.state = None

    def config(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]


class _FakeLabel:
    def __init__(self):
        self.text = ""

    def config(self, **kwargs):
        if "text" in kwargs:
            self.text = kwargs["text"]


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


class TestInterfaceLifecycleUnit(unittest.TestCase):
    def setUp(self):
        if MarsColonyInterface is None:
            self.skipTest(f"GUI dependency unavailable in test environment: {IMPORT_ERROR}")
        self.app = MarsColonyInterface.__new__(MarsColonyInterface)
        self.app.STATE_IDLE = MarsColonyInterface.STATE_IDLE
        self.app.STATE_RUNNING = MarsColonyInterface.STATE_RUNNING
        self.app.STATE_PAUSED = MarsColonyInterface.STATE_PAUSED
        self.app.STATE_STOPPED = MarsColonyInterface.STATE_STOPPED
        self.app.STATE_STARTING = MarsColonyInterface.STATE_STARTING
        self.app.run_state = self.app.STATE_IDLE
        self.app.sim = None
        self.app.base_dt = 0.1
        self.app._run_loop_job = None
        self.app.start_button = _FakeButton()
        self.app.run_batch_button = _FakeButton()
        self.app.pause_button = _FakeButton()
        self.app.stop_button = _FakeButton()
        self.app.lifecycle_label = _FakeLabel()
        self.app._startup_status_var = None
        self.app._startup_progress = None
        self.app._startup_progressbar = None
        self.app._startup_queue = None
        self.app._startup_worker = None

        self.applied = 0
        self.app.apply_experiment_settings = self._apply
        self.app._begin_async_startup = self._begin_async_startup
        self.app._cancel_run_loop = lambda: None
        self.app._close_startup_dialog = lambda: None
        self.app._finalize_simulation_install = lambda sim: setattr(self.app, "sim", sim)
        self.app._schedule_next_tick = lambda: None
        self.startup_invocations = 0
        self.error_messages = []

    def _apply(self):
        self.applied += 1
        self.app.sim = _FakeSim()
        self.app.run_state = self.app.STATE_IDLE

    def _begin_async_startup(self):
        self.startup_invocations += 1
        self.app.run_state = self.app.STATE_STARTING

    def test_start_pause_resume_stop_transitions(self):
        scheduled = []
        self.app._schedule_next_tick = lambda: scheduled.append("tick")

        self.app.start_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_STARTING)
        self.assertEqual(self.startup_invocations, 1)
        self.assertEqual(self.applied, 0)
        self.assertEqual(scheduled, [])

        self.app._finalize_startup_success(_FakeSim())
        self.assertEqual(self.app.run_state, self.app.STATE_RUNNING)

        self.app.pause_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_PAUSED)

        before_time = self.app.sim.time
        self.app._run_loop_tick()
        self.assertEqual(self.app.sim.time, before_time)

        self.app.start_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_RUNNING)
        self.assertEqual(self.startup_invocations, 1, "Resume should not reinitialize simulation")

        self.app.stop_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_STOPPED)
        self.assertEqual(self.app.sim.stop_calls, 1)

    def test_tick_advances_and_refreshes_views(self):
        self.app.sim = _FakeSim()
        self.app.run_state = self.app.STATE_RUNNING

        calls = {"env": 0, "agents": 0, "events": 0, "dashboard": 0, "construction": 0, "schedule": 0}
        self.app.update_environment_plot = lambda frame=None: calls.__setitem__("env", calls["env"] + 1)
        self.app.update_agent_table = lambda: calls.__setitem__("agents", calls["agents"] + 1)
        self.app.update_event_monitor = lambda: calls.__setitem__("events", calls["events"] + 1)
        self.app.update_dashboard = lambda: calls.__setitem__("dashboard", calls["dashboard"] + 1)
        self.app._sync_construction_summaries = lambda: calls.__setitem__("construction", calls["construction"] + 1)
        self.app._schedule_next_tick = lambda: calls.__setitem__("schedule", calls["schedule"] + 1)

        self.app._run_loop_tick()

        self.assertEqual(self.app.sim.update_calls, 1)
        self.assertAlmostEqual(self.app.sim.time, 0.1)
        self.assertEqual(calls, {"env": 1, "agents": 1, "events": 1, "dashboard": 1, "construction": 1, "schedule": 1})

    def test_start_is_ignored_while_startup_already_in_progress(self):
        self.app.run_state = self.app.STATE_STARTING
        self.app.start_experiment()
        self.assertEqual(self.startup_invocations, 0)

    def test_queue_events_finalize_success_and_failure_paths(self):
        sim = _FakeSim()
        self.app.run_state = self.app.STATE_STARTING
        self.app._handle_startup_event({"type": "startup_success", "sim": sim})
        self.assertIs(self.app.sim, sim)
        self.assertEqual(self.app.run_state, self.app.STATE_RUNNING)

        self.app.run_state = self.app.STATE_STARTING
        with patch("interface.messagebox.showerror", side_effect=lambda _t, m: self.error_messages.append(m)):
            self.app._handle_startup_event({"type": "startup_failure", "error_message": "boom"})
        self.assertEqual(self.app.run_state, self.app.STATE_IDLE)
        self.assertIsNone(self.app.sim)
        self.assertIn("boom", self.error_messages[0])

    def test_finalize_simulation_install_populates_startup_views_in_expected_order(self):
        calls = []
        self.app._cancel_run_loop = lambda: calls.append("_cancel_run_loop")
        self.app._update_control_states = lambda: calls.append("_update_control_states")
        self.app.update_environment_plot = lambda: calls.append("update_environment_plot")
        self.app.update_agent_table = lambda: calls.append("update_agent_table")
        self.app.update_event_monitor = lambda: calls.append("update_event_monitor")
        self.app.update_dashboard = lambda: calls.append("update_dashboard")
        self.app._sync_construction_summaries = lambda: calls.append("_sync_construction_summaries")
        self.app._update_system_log = lambda: calls.append("_update_system_log")
        self.app._update_backend_status_display = lambda: calls.append("_update_backend_status_display")
        self.app._update_observability_status_display = lambda: calls.append("_update_observability_status_display")
        self.app.update_interaction_tab = lambda: calls.append("update_interaction_tab")
        self.app.update_brain_tab = lambda: calls.append("update_brain_tab")
        self.app.update_dik_derivations_tab = lambda: calls.append("update_dik_derivations_tab")
        self.app.update_rule_derivations_tab = lambda: calls.append("update_rule_derivations_tab")
        sim = _FakeSim()

        MarsColonyInterface._finalize_simulation_install(self.app, sim)

        self.assertIs(self.app.sim, sim)
        self.assertEqual(self.app.run_state, self.app.STATE_IDLE)
        self.assertEqual(
            calls,
            [
                "_cancel_run_loop",
                "_update_control_states",
                "update_environment_plot",
                "update_agent_table",
                "update_event_monitor",
                "update_dashboard",
                "_sync_construction_summaries",
                "_update_system_log",
                "_update_backend_status_display",
                "_update_observability_status_display",
                "update_interaction_tab",
                "update_brain_tab",
                "update_dik_derivations_tab",
                "update_rule_derivations_tab",
            ],
        )


if __name__ == "__main__":
    unittest.main()
