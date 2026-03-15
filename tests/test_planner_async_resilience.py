import json
import tempfile
import time
import unittest
from unittest.mock import patch

from modules.simulation import SimulationState


class TestPlannerAsyncResilience(unittest.TestCase):
    def _build_sim(self, tmpdir):
        return SimulationState(
            phases=[],
            project_root=tmpdir,
            brain_backend="local_http",
            brain_backend_options={"timeout_s": 0.01, "max_retries": 0, "fallback_backend": "rule_brain"},
            planner_config={
                "planner_interval_steps": 1,
                "planner_interval_time": 0.0,
                "planner_timeout_seconds": 0.05,
                "degraded_consecutive_failures_threshold": 2,
                "degraded_cooldown_seconds": 0.3,
            },
        )

    def test_non_blocking_update_with_slow_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)
            for agent in sim.agents:
                agent.planner_cadence.planner_timeout_seconds = 5.0

            def slow_generate(self, request_packet):
                time.sleep(0.25)
                return self.fallback.generate_plan(request_packet)

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=slow_generate):
                started = time.perf_counter()
                sim.update(0.1)
                elapsed = time.perf_counter() - started

            self.assertLess(elapsed, 0.5)
            self.assertGreater(sim.time, 0.0)
            sim.stop()

    def test_inflight_suppression_skips_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)
            agent = sim.agents[0]
            agent.planner_state["status"] = "in_flight"
            agent.planner_state["request_id"] = "unit-inflight"
            started_before = agent.planner_state.get("total_started", 0)

            sim.update(0.1)

            started_after = agent.planner_state.get("total_started", 0)
            self.assertEqual(started_after, started_before)
            events = sim.logger.get_recent_events(120)
            skipped = [e for e in events if e.get("event_type") == "planner_request_skipped_inflight"]
            self.assertTrue(skipped)
            sim.stop()

    def test_timeout_degradation_and_cooldown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)

            def very_slow_generate(self, request_packet):
                time.sleep(0.35)
                return self.fallback.generate_plan(request_packet)

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=very_slow_generate):
                for _ in range(12):
                    sim.update(0.1)
                events = sim.logger.get_recent_events(250)
                degraded_events = [e for e in events if e.get("event_type") == "backend_degraded_mode_started"]
                cooldown_events = [e for e in events if e.get("event_type") == "planner_request_skipped_cooldown"]
                self.assertTrue(degraded_events)
                self.assertTrue(cooldown_events)
            sim.stop()

    def test_late_response_is_marked_stale_after_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)

            def very_slow_generate(self, request_packet):
                time.sleep(0.2)
                return self.fallback.generate_plan(request_packet)

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=very_slow_generate):
                sim.update(0.1)
                for _ in range(3):
                    sim.update(0.1)
                time.sleep(0.25)
                sim.update(0.1)
                events = sim.logger.get_recent_events(250)
                stale_events = [e for e in events if e.get("event_type") == "planner_request_result_arrived_stale"]
                self.assertTrue(stale_events)
            sim.stop()

    def test_run_summary_contains_planner_responsiveness_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)
            sim.update(0.2)
            sim.stop()
            session_dir = next((sim.logger.output_session.outputs_root).iterdir())
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            planner_metrics = run_summary["process"].get("planner_responsiveness", {})
            self.assertIn("requests_started", planner_metrics)
            self.assertIn("requests_skipped_due_to_inflight", planner_metrics)
            self.assertIn("ui_safe_fallback_count", planner_metrics)


if __name__ == "__main__":
    unittest.main()
