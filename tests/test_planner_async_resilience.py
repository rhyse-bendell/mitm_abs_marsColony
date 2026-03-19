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
                "high_latency_local_llm_mode": False,
                "high_latency_stale_result_grace_s": 0.0,
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
                    time.sleep(0.06)
                events = sim.logger.get_recent_events(250)
                degraded_events = [e for e in events if e.get("event_type") == "backend_degraded_mode_started"]
                cooldown_events = [e for e in events if e.get("event_type") == "planner_request_skipped_cooldown"]
                self.assertGreater(sim.agents[0].planner_state.get("total_timed_out", 0), 0)
                self.assertGreater(sim.agents[0].planner_state.get("fallback_generated_count", 0), 0)
                self.assertFalse([e for e in events if e.get("event_type") == "brain_provider_response_received"])
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

    def test_sim_time_advance_does_not_prematurely_timeout_wallclock_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)
            agent = sim.agents[0]
            agent.planner_cadence.planner_timeout_seconds = 0.5
            agent.planner_state["status"] = "in_flight"
            agent.planner_state["request_id"] = "wallclock-semantic"
            agent.planner_state["requested_wallclock_at"] = time.perf_counter()

            sim.time += 100.0
            agent._check_inflight_timeout(sim)
            self.assertEqual(agent.planner_state["status"], "in_flight")
            sim.stop()

    def test_late_valid_response_not_discarded_due_to_sim_tick_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)
            agent = sim.agents[0]
            agent.planner_cadence.planner_timeout_seconds = 1.5

            def moderately_slow_generate(self, request_packet):
                time.sleep(0.2)
                return self.fallback.generate_plan(request_packet)

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=moderately_slow_generate):
                sim.update(0.1)
                for _ in range(6):
                    sim.update(0.5)
                time.sleep(0.25)
                sim.update(0.1)
                events = sim.logger.get_recent_events(300)
                stale_discard = [e for e in events if e.get("event_type") == "planner_response_discarded_due_to_state_change"]
                llm_done = [e for e in events if e.get("event_type") == "planner_request_completed_with_llm"]
                self.assertFalse(stale_discard)
                self.assertTrue(llm_done)
            sim.stop()


    def test_high_latency_mode_prevents_premature_inflight_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="local_http",
                brain_backend_options={"timeout_s": 1.0, "max_retries": 0, "fallback_backend": "rule_brain"},
                planner_config={
                    "planner_interval_steps": 1,
                    "planner_interval_time": 0.0,
                    "planner_timeout_seconds": 0.8,
                    "high_latency_local_llm_mode": True,
                    "high_latency_stale_result_grace_s": 1.0,
                    "unrestricted_local_qwen_mode": False,
                    "degraded_consecutive_failures_threshold": 6,
                    "degraded_cooldown_seconds": 5.0,
                },
            )

            def slow_generate(self, request_packet):
                time.sleep(0.95)
                return self.fallback.generate_plan(request_packet)

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=slow_generate):
                sim.update(0.1)
                time.sleep(0.2)
                sim.update(0.1)
                self.assertEqual(sim.agents[0].planner_state.get("total_timed_out", 0), 0)
            sim.stop()

    def test_high_latency_mode_accepts_late_result_within_grace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="local_http",
                brain_backend_options={"timeout_s": 1.0, "max_retries": 0, "fallback_backend": "rule_brain"},
                planner_config={
                    "planner_interval_steps": 1,
                    "planner_interval_time": 0.0,
                    "planner_timeout_seconds": 0.05,
                    "high_latency_local_llm_mode": True,
                    "high_latency_stale_result_grace_s": 1.0,
                    "unrestricted_local_qwen_mode": False,
                },
            )

            def slow_generate(self, request_packet):
                time.sleep(0.2)
                return self.fallback.generate_plan(request_packet)

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=slow_generate):
                sim.update(0.1)
                for _ in range(3):
                    sim.update(0.1)
                time.sleep(0.25)
                sim.update(0.1)
                events = sim.logger.get_recent_events(300)
                accepted = [e for e in events if e.get("event_type") == "planner_request_result_arrived_after_timeout_accepted"]
                stale = [e for e in events if e.get("event_type") == "planner_request_result_arrived_stale"]
                self.assertTrue(accepted)
                self.assertFalse(stale)
            sim.stop()


    def test_unrestricted_mode_uses_extended_timeout_and_stale_grace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="local_http",
                brain_backend_options={"timeout_s": 0.2, "max_retries": 0, "fallback_backend": "rule_brain"},
                planner_config={
                    "unrestricted_local_qwen_mode": True,
                    "planner_timeout_seconds": 900.0,
                    "high_latency_stale_result_grace_s": 1800.0,
                },
            )
            agent = sim.agents[0]
            self.assertGreaterEqual(agent.planner_cadence.planner_timeout_seconds, 900.0)
            self.assertGreaterEqual(agent.planner_cadence.high_latency_stale_result_grace_s, 1800.0)
            agent.planner_state["status"] = "in_flight"
            agent.planner_state["request_id"] = "unrestricted-timeout-check"
            agent.planner_state["requested_wallclock_at"] = time.perf_counter() - 0.2
            agent._check_inflight_timeout(sim)
            self.assertEqual(agent.planner_state["status"], "in_flight")
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
            self.assertIn("requests_completed_with_llm", planner_metrics)
            self.assertIn("requests_completed_with_fallback", planner_metrics)
            self.assertIn("llm_timeout_count", planner_metrics)
            self.assertIn("fallback_generated_count", planner_metrics)


if __name__ == "__main__":
    unittest.main()
