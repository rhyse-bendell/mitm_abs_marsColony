"""Observer GUI hosting a realtime human-participant session (skipped when Tk is unavailable)."""
import contextlib
import importlib
import io
import tempfile
import time
import tkinter as tk
import unittest
from unittest import mock


class TestObserverGuiWithHumanParticipant(unittest.TestCase):
    def setUp(self):
        try:
            self.mod = importlib.import_module("interface")
            self.app = self.mod.MarsColonyInterface()
        except (ModuleNotFoundError, tk.TclError) as exc:
            self.skipTest(f"GUI unavailable: {exc}")
            return
        self.app.root.withdraw()

    def tearDown(self):
        if hasattr(self, "app"):
            with contextlib.suppress(Exception):
                self.app._close_human_session()
            self.app.root.destroy()

    def test_controller_selector_offers_human_participant_per_role_and_resolves_it(self):
        self.app.agent_controller_vars["Architect"].set("Human Participant")
        assignments = self.app.controller_assignments()
        self.assertEqual(assignments["Architect"]["controller_type"], "human_participant")
        self.assertEqual(assignments["Botanist"]["controller_type"], "procedural_baseline")
        self.assertEqual(assignments["Engineer"]["controller_type"], "procedural_baseline")

    def test_session_hosting_pause_stop_and_controller_labels(self):
        from modules.simulation import SimulationState

        self.app.agent_controller_vars["Architect"].set("Human Participant")
        with contextlib.redirect_stdout(io.StringIO()):
            sim = SimulationState(speed=1.0, seed=3, outputs_root=tempfile.mkdtemp(), controller_assignments=self.app.controller_assignments())
        with mock.patch.object(self.mod.messagebox, "showinfo") as info, mock.patch("webbrowser.open") as opened:
            self.app._finalize_simulation_install(sim)
        self.assertIsNotNone(self.app.human_session)
        self.assertTrue(info.called and opened.called and "Architect" in self.app.human_session.urls())
        self.assertTrue(self.app.human_session.scheduler.paused)  # the mission clock has not started
        # Agent States panels are labelled with each agent's controller
        titles = {p["panel"].cget("text") for p in self.app.agent_state_panels.values()}
        self.assertTrue(any("[Human Participant]" in t and "Architect" in t for t in titles), titles)
        self.assertTrue(any("[Procedural Baseline]" in t and "Botanist" in t for t in titles), titles)
        # run: wall-clock paced ticks through the Tk loop
        self.app.run_state = self.app.STATE_RUNNING
        self.app.human_session.resume()
        self.app._schedule_next_tick = lambda: None
        with contextlib.redirect_stdout(io.StringIO()):
            end = time.monotonic() + 1.2
            while time.monotonic() < end:
                self.app._run_loop_tick()
                time.sleep(0.05)
        self.assertGreater(sim.time, 0.8)
        self.assertLess(sim.time, 2.5)  # ~1x wall clock, not the stepped 'speed x base_dt' of simulation mode
        self.app.pause_experiment()
        frozen = sim.time
        time.sleep(0.4)
        with contextlib.redirect_stdout(io.StringIO()):
            self.app._run_loop_tick()
        self.assertEqual(sim.time, frozen)
        stops = []
        real_stop = sim.stop
        sim.stop = lambda: (stops.append(1), real_stop())
        with contextlib.redirect_stdout(io.StringIO()):
            self.app.stop_experiment()
        self.assertEqual(len(stops), 1)
        self.assertIsNone(self.app.human_session)


if __name__ == "__main__":
    unittest.main()
