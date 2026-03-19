import tempfile
import time
import unittest
from unittest.mock import patch

from modules.simulation import SimulationState


class TestFallbackBootstrapMode(unittest.TestCase):
    def _run_steps(self, sim, steps=40, dt=0.2):
        for _ in range(steps):
            sim.update(dt)
            time.sleep(0.03)

    def test_startup_sanity_failure_forces_bootstrap_sequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity._post_chat_completion", side_effect=TimeoutError("forced startup timeout")):
                sim = SimulationState(
                    phases=[],
                    project_root=tmpdir,
                    flash_mode=True,
                    brain_backend="local_http",
                    brain_backend_options={"timeout_s": 0.1, "max_retries": 0, "fallback_backend": "rule_brain"},
                    planner_config={"planner_interval_steps": 1, "planner_interval_time": 0.0, "planner_timeout_seconds": 0.2},
                )
            with patch("modules.agent.random.random", return_value=0.0):
                self._run_steps(sim, steps=90, dt=0.2)
            audit = sim.runtime_witness_audit.finalize()
            left_spawn = {agent.name: bool(agent.startup_state.get("left_spawn")) for agent in sim.agents}
            self.assertTrue(all(left_spawn.values()))

            first_failures = audit.get("summary", {}).get("first_failed_step_categories", {})
            self.assertNotEqual(first_failures.get("category"), "source_not_accessed")
            for agent in sim.agents:
                self.assertEqual(agent.fallback_bootstrap.get("activation_reason"), "startup_sanity_failed")
                self.assertEqual(agent.fallback_bootstrap.get("last_forced_action"), "inspect_information_source")
            sim.stop()

    def test_repeated_runtime_fallback_activates_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                flash_mode=True,
                brain_backend="local_http",
                brain_backend_options={"timeout_s": 0.1, "max_retries": 0, "fallback_backend": "rule_brain"},
                planner_config={
                    "planner_interval_steps": 1,
                    "planner_interval_time": 0.0,
                    "planner_timeout_seconds": 0.2,
                    "enable_startup_llm_sanity": False,
                },
            )

            def forced_fallback(self, request_packet):
                response = self.fallback.generate_plan(request_packet)
                self.last_outcome = {
                    "fallback": True,
                    "reason": "forced-test-fallback",
                    "latency_ms": 1.0,
                    "result_source": "fallback_safe_policy",
                    "outcome_category": "llm_error_with_fallback",
                }
                self.last_trace = {"fallback_used": True, "result_source": "fallback_safe_policy"}
                return response

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=forced_fallback), patch(
                "modules.agent.random.random", return_value=0.0
            ):
                self._run_steps(sim, steps=40, dt=0.2)
            self.assertTrue(any(int(a.fallback_bootstrap.get("runtime_fallback_triggers", 0)) >= 2 for a in sim.agents))
            self.assertTrue(any(a.fallback_bootstrap.get("activation_reason") == "runtime_fallback_repeated" for a in sim.agents))
            self.assertTrue(any(a.fallback_bootstrap.get("last_forced_action") == "inspect_information_source" for a in sim.agents))
            sim.stop()


if __name__ == "__main__":
    unittest.main()
