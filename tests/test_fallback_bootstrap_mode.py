import tempfile
import time
import unittest
from unittest.mock import patch

from modules.brain_provider import OllamaLocalBrainProvider
from modules.simulation import SimulationState


class TestFallbackBootstrapMode(unittest.TestCase):
    def _run_steps(self, sim, steps=40, dt=0.2):
        for _ in range(steps):
            sim.update(dt)
            time.sleep(0.03)

    def test_runtime_fallback_run_remains_productive_without_startup_sanity_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                flash_mode=True,
                brain_backend="local_http",
                brain_backend_options={"timeout_s": 0.1, "max_retries": 0, "fallback_backend": "rule_brain"},
                planner_config={"planner_interval_steps": 1, "planner_interval_time": 0.0, "planner_timeout_seconds": 0.2},
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
                self._run_steps(sim, steps=90, dt=0.2)
            audit = sim.runtime_witness_audit.finalize()
            self.assertGreaterEqual(float(sim.time), 1.0)

            first_failures = audit.get("summary", {}).get("first_failed_step_categories", {})
            self.assertNotEqual(first_failures.get("category"), "source_not_accessed")
            for agent in sim.agents:
                self.assertNotEqual(agent.fallback_bootstrap.get("activation_reason"), "startup_sanity_failed")
                runtime = sim.get_agent_brain_runtime(agent)
                self.assertFalse(runtime.get("hard_demoted"))
                self.assertIsInstance(runtime.get("provider"), OllamaLocalBrainProvider)
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
            self.assertGreaterEqual(sim.backend_fallback_count, 1)
            self.assertTrue(all(not bool(sim.get_agent_brain_runtime(a).get("hard_demoted")) for a in sim.agents))
            self.assertTrue(all(isinstance(sim.get_agent_brain_runtime(a).get("provider"), OllamaLocalBrainProvider) for a in sim.agents))
            sim.stop()

    def test_bootstrap_stages_shared_then_role_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = next(a for a in sim.agents if a.role == "Architect")
            agent.activate_fallback_bootstrap(sim_state=sim, reason="test_stage_transition")
            agent._ensure_source_state(sim.environment)
            agent.startup_state["left_spawn"] = True

            # Shared packet has been consumed already and is exhausted for this agent.
            agent.source_inspection_state["Team_Info"] = "revisitable_due_to_gap"
            agent.source_exhaustion_state["Team_Info"]["exhausted"] = True

            forced = agent._bootstrap_override_decision(sim.environment, sim_state=sim)
            self.assertIsNotNone(forced)
            self.assertEqual(forced.target_id, "Architect_Info")
            self.assertEqual(agent.fallback_bootstrap.get("stage"), "role")

            # Once role packet is completed/exhausted, bootstrap can complete.
            agent.source_inspection_state["Architect_Info"] = "inspected"
            agent.source_exhaustion_state["Architect_Info"]["exhausted"] = True
            followup = agent._bootstrap_override_decision(sim.environment, sim_state=sim)
            self.assertIsNone(followup)
            self.assertFalse(agent.fallback_bootstrap.get("active"))
            sim.stop()

    def test_forced_llm_failure_opt_in_sticky_hard_demotion_keeps_run_productive(self):
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
                    "sticky_backend_demotion_enabled": True,
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

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=forced_fallback), patch("modules.agent.random.random", return_value=0.0):
                self._run_steps(sim, steps=110, dt=0.2)

            for agent in sim.agents:
                runtime = sim.get_agent_brain_runtime(agent)
                self.assertFalse(runtime.get("hard_demoted"))
                self.assertLess(int(agent.planner_state.get("total_skipped_inflight", 0)), 2)
                self.assertGreater(
                    int(agent.planner_state.get("productive_fallback_action_count", 0))
                    + int(agent.planner_state.get("requests_completed_with_fallback", 0))
                    + int(agent.planner_state.get("requests_completed_with_llm", 0)),
                    0,
                )
            sim.stop()


if __name__ == "__main__":
    unittest.main()
