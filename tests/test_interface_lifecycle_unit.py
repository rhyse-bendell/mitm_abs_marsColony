import unittest

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
        self.app.run_state = self.app.STATE_IDLE
        self.app.sim = None
        self.app.base_dt = 0.1
        self.app._run_loop_job = None
        self.app.start_button = _FakeButton()
        self.app.pause_button = _FakeButton()
        self.app.stop_button = _FakeButton()
        self.app.lifecycle_label = _FakeLabel()

        self.applied = 0
        self.app.apply_experiment_settings = self._apply
        self.app._cancel_run_loop = lambda: None

    def _apply(self):
        self.applied += 1
        self.app.sim = _FakeSim()
        self.app.run_state = self.app.STATE_IDLE

    def test_start_pause_resume_stop_transitions(self):
        scheduled = []
        self.app._schedule_next_tick = lambda: scheduled.append("tick")

        self.app.start_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_RUNNING)
        self.assertEqual(self.applied, 1)
        self.assertEqual(scheduled, ["tick"])

        self.app.pause_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_PAUSED)

        before_time = self.app.sim.time
        self.app._run_loop_tick()
        self.assertEqual(self.app.sim.time, before_time)

        self.app.start_experiment()
        self.assertEqual(self.app.run_state, self.app.STATE_RUNNING)
        self.assertEqual(self.applied, 1, "Resume should not reinitialize simulation")

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


if __name__ == "__main__":
    unittest.main()
