import tempfile
import unittest

from modules.simulation import SimulationState

try:
    from interface import MarsColonyInterface
except ModuleNotFoundError:
    MarsColonyInterface = None


class TestGuiObservabilityHelpers(unittest.TestCase):
    def test_format_elapsed_duration(self):
        if MarsColonyInterface is None:
            self.skipTest("Interface dependencies unavailable")
        self.assertEqual(MarsColonyInterface._format_elapsed_duration(0.0), "00:00:00.0")
        self.assertEqual(MarsColonyInterface._format_elapsed_duration(65.3), "00:01:05.3")
        self.assertEqual(MarsColonyInterface._format_elapsed_duration(3661.2), "01:01:01.2")

    def test_barrier_summary_text(self):
        if MarsColonyInterface is None:
            self.skipTest("Interface dependencies unavailable")
        self.assertEqual(MarsColonyInterface._format_barrier_summary({"barrier_active": False}), "Barrier: none")
        self.assertEqual(
            MarsColonyInterface._format_barrier_summary(
                {
                    "barrier_active": True,
                    "blocking_request_ids": ["r1", "r2", "r3"],
                    "blocking_agent_ids": ["a1", "a2"],
                }
            ),
            "Barrier: active (3 requests, 2 agents)",
        )


class TestSimulationObservabilityStatus(unittest.TestCase):
    def test_current_pause_elapsed_updates_while_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="rule_brain")
            sim.planner_barrier_state.update(
                {
                    "active": True,
                    "pause_started_wallclock_at": 10.0,
                    "total_wallclock_wait_s": 1.5,
                }
            )
            status_1 = sim.get_observability_status(now_wallclock=12.0)
            status_2 = sim.get_observability_status(now_wallclock=15.0)
            self.assertAlmostEqual(status_1["current_cognition_pause_elapsed_s"], 2.0)
            self.assertAlmostEqual(status_2["current_cognition_pause_elapsed_s"], 5.0)
            self.assertAlmostEqual(status_1["cumulative_cognition_wait_s"], 3.5)
            self.assertAlmostEqual(status_2["cumulative_cognition_wait_s"], 6.5)
            sim.stop()

    def test_cumulative_wait_across_multiple_pauses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="rule_brain")
            sim.planner_barrier_state["total_wallclock_wait_s"] = 2.5
            sim.planner_barrier_state["active"] = True
            sim.planner_barrier_state["pause_started_wallclock_at"] = 20.0
            status = sim.get_observability_status(now_wallclock=23.0)
            self.assertAlmostEqual(status["current_cognition_pause_elapsed_s"], 3.0)
            self.assertAlmostEqual(status["cumulative_cognition_wait_s"], 5.5)
            sim.stop()

    def test_run_stop_preserves_wallclock_values_until_new_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="rule_brain")
            sim.run_started_wallclock_at = 100.0
            sim.run_stopped_wallclock_at = 103.0
            status = sim.get_observability_status(now_wallclock=110.0)
            self.assertAlmostEqual(status["run_wallclock_elapsed_s"], 3.0)
            sim.stop()

            sim2 = SimulationState(phases=[], project_root=tmpdir, brain_backend="rule_brain")
            sim2.run_started_wallclock_at = 200.0
            status2 = sim2.get_observability_status(now_wallclock=201.0)
            self.assertAlmostEqual(status2["run_wallclock_elapsed_s"], 1.0)
            self.assertAlmostEqual(status2["cumulative_cognition_wait_s"], 0.0)
            self.assertFalse(status2["barrier_active"])
            sim2.stop()


if __name__ == "__main__":
    unittest.main()
