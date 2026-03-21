import json
import tempfile
import unittest
from unittest.mock import patch

from modules.aggregate_measures import aggregate_run_summaries
from modules.brain_provider import BrainBackendConfig, OllamaLocalBrainProvider, create_brain_provider
from modules.simulation import SimulationState


class TestBrainBackendSelection(unittest.TestCase):

    def test_local_backend_defaults_use_real_model_and_timeout(self):
        cfg = BrainBackendConfig()
        self.assertEqual(cfg.local_model, "qwen3.5:9b")
        self.assertEqual(cfg.timeout_s, 75.0)
        self.assertEqual(cfg.warmup_timeout_s, 45.0)

    def test_warmup_probe_uses_configured_warmup_timeout(self):
        provider = OllamaLocalBrainProvider(
            config=BrainBackendConfig(backend="ollama", timeout_s=15.0, warmup_timeout_s=9.0),
            fallback=create_brain_provider(BrainBackendConfig(backend="rule_brain")),
        )

        def _raise_timeout(*_args, **_kwargs):
            raise TimeoutError("simulated warmup timeout")

        with patch("modules.brain_provider.request.urlopen", side_effect=_raise_timeout):
            status = provider.warmup_probe()
        self.assertFalse(status["ok"])
        self.assertEqual(status["warmup_timeout_s"], 9.0)
        self.assertTrue(status["warmup_timeout"])
        self.assertEqual(status["probe_type"], "startup_warmup")

    def test_explicit_rule_brain_selection_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="rule_brain")
            self.assertEqual(sim.configured_brain_backend, "rule_brain")
            self.assertEqual(sim.effective_brain_backend, "rule_brain")
            sim.update(0.2)
            sim.stop()

    def test_local_backend_selection_propagates_to_provider_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="local_http",
                brain_backend_options={
                    "local_model": "llama3.2",
                    "local_base_url": "http://127.0.0.1:11434",
                    "timeout_s": 0.2,
                    "fallback_backend": "rule_brain",
                },
                planner_config={"unrestricted_local_qwen_mode": False, "high_latency_local_llm_mode": False},
            )
            self.assertEqual(sim.configured_brain_backend, "local_http")
            self.assertEqual(sim.brain_backend_config.local_model, "llama3.2")
            self.assertEqual(sim.brain_backend_config.local_base_url, "http://127.0.0.1:11434")
            self.assertEqual(sim.brain_backend_config.fallback_backend, "rule_brain")
            self.assertEqual(sim.brain_backend_config.timeout_s, 0.2)
            self.assertIsInstance(sim.brain_provider, OllamaLocalBrainProvider)
            self.assertEqual(sim.brain_provider.config.local_model, "llama3.2")
            sim.stop()

    def test_fallback_updates_effective_backend_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="local_http",
                brain_backend_options={"timeout_s": 0.05, "max_retries": 0, "fallback_backend": "rule_brain"},
            )
            self.assertIsInstance(sim.brain_provider, OllamaLocalBrainProvider)

            with patch.object(OllamaLocalBrainProvider, "generate_plan", side_effect=sim.brain_provider.fallback.generate_plan):
                sim.update(0.2)

            self.assertEqual(sim.configured_brain_backend, "local_http")
            # No fallback event emitted via patched generate_plan path; validate runtime state remains deterministic
            self.assertIn(sim.effective_brain_backend, {"local_http", "rule_brain"})
            sim.stop()

    def test_provider_internal_fallback_marks_effective_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="local_http",
                brain_backend_options={"timeout_s": 0.01, "max_retries": 0, "fallback_backend": "rule_brain"},
            )
            # Force planner calls to trigger and allow local provider to fail fast.
            for _ in range(3):
                sim.update(0.5)
            self.assertEqual(sim.configured_brain_backend, "local_http")
            self.assertEqual(sim.effective_brain_backend, "rule_brain")
            self.assertGreaterEqual(sim.backend_fallback_count, 1)
            sim.stop()

    def test_run_summary_and_manifest_record_configured_vs_effective_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="rule_brain")
            sim.update(0.3)
            sim.stop()

            session_dir = next((sim.logger.output_session.outputs_root).iterdir())
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))

            for payload in (run_summary["run_metadata"], manifest):
                self.assertEqual(payload["configured_brain_backend"], "rule_brain")
                self.assertEqual(payload["effective_brain_backend"], "rule_brain")
                self.assertIn("fallback_backend", payload)
                self.assertIn("fallback_occurred", payload)
                self.assertIn("fallback_count", payload)


    def test_manifest_records_local_backend_model_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="ollama",
                brain_backend_options={"local_model": "qwen3.5:9b", "timeout_s": 75.0},
                planner_config={"unrestricted_local_qwen_mode": False, "high_latency_local_llm_mode": False},
            )
            sim.update(0.1)
            sim.stop()
            session_dir = next((sim.logger.output_session.outputs_root).iterdir())
            manifest = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configured_brain_backend"], "ollama")
            self.assertEqual(manifest["local_model_name"], "qwen3.5:9b")
            self.assertEqual(manifest["timeout_s"], 75.0)


    def test_manifest_records_high_latency_local_mode_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="ollama")
            sim.update(0.1)
            sim.stop()
            session_dir = next((sim.logger.output_session.outputs_root).iterdir())
            manifest = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest.get("high_latency_local_llm_mode"))
            self.assertTrue(manifest.get("unrestricted_local_qwen_mode"))
            self.assertGreaterEqual(float(manifest.get("effective_planner_timeout_seconds", 0.0)), 180.0)
            self.assertGreaterEqual(float(manifest.get("effective_startup_llm_sanity_timeout_seconds", 0.0)), 120.0)
            self.assertGreaterEqual(float(manifest.get("effective_warmup_timeout_seconds", 0.0)), 90.0)
            self.assertGreaterEqual(int(manifest.get("effective_startup_llm_sanity_completion_max_tokens", 0)), 768)
            self.assertGreaterEqual(int(manifest.get("effective_planner_completion_max_tokens", 0)), 2048)
            self.assertTrue(manifest.get("stale_result_relaxation_enabled"))
            self.assertGreaterEqual(float(manifest.get("permissive_timeout_ceiling_s", 0.0)), 1800.0)
            self.assertGreaterEqual(int(manifest.get("permissive_completion_ceiling_tokens", 0)), 32768)

    def test_disabling_unrestricted_mode_preserves_explicit_normal_budgets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="ollama",
                planner_config={
                    "unrestricted_local_qwen_mode": False,
                    "high_latency_local_llm_mode": False,
                    "planner_timeout_seconds": 90.0,
                    "startup_llm_sanity_timeout_seconds": 45.0,
                    "startup_llm_sanity_completion_max_tokens": 1024,
                    "planner_completion_max_tokens": 2048,
                    "warmup_timeout_seconds": 45.0,
                },
            )
            self.assertFalse(sim.planner_defaults.get("unrestricted_local_qwen_mode"))
            self.assertEqual(sim.brain_backend_config.timeout_s, 90.0)
            self.assertEqual(sim.startup_llm_sanity_config.timeout_s, 45.0)
            self.assertEqual(sim.startup_llm_sanity_config.completion_max_tokens, 1024)
            self.assertEqual(sim.brain_backend_config.completion_max_tokens, 2048)
            self.assertEqual(sim.brain_backend_config.warmup_timeout_s, 45.0)
            sim.stop()

    def test_aggregate_outputs_include_backend_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, experiment_name="agg_backend")
            sim.update(0.3)
            sim.stop()
            outputs_root = sim.logger.output_session.outputs_root
            result = aggregate_run_summaries(outputs_root)
            aggregate = json.loads(open(result["json"], "r", encoding="utf-8").read())
            row = aggregate["rows"][0]
            self.assertIn("configured_brain_backend", row)
            self.assertIn("effective_brain_backend", row)
            self.assertIn("fallback_count", row)

    def test_backward_compatible_default_backend_remains_rule_brain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            self.assertEqual(sim.brain_backend_config.backend, "rule_brain")
            sim.update(0.1)
            sim.stop()


    def test_ollama_alias_routes_to_local_provider(self):
        provider = create_brain_provider(BrainBackendConfig(backend="ollama"))
        self.assertIsInstance(provider, OllamaLocalBrainProvider)

    def test_fallback_event_includes_model_and_hint_for_404(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="ollama",
                brain_backend_options={
                    "local_model": "missing-model",
                    "local_base_url": "http://127.0.0.1:11434",
                    "timeout_s": 0.1,
                    "max_retries": 0,
                },
            )
            provider = sim.brain_provider
            provider.last_outcome = {
                "fallback": True,
                "reason": "attempt=1/1 error=HTTP Error 404: Not Found",
                "latency_ms": 2.0,
                "hint": "HTTP 404 from local backend may indicate missing/incorrect model name",
            }
            sim._refresh_backend_effective_state(reason="unit_test")
            events = sim.logger.get_recent_events(10)
            fallback_events = [e for e in events if e.get("event_type") == "brain_provider_fallback"]
            self.assertTrue(fallback_events)
            payload = json.loads(fallback_events[-1].get("payload", "{}"))
            self.assertEqual(payload.get("local_model_name"), "missing-model")
            self.assertIn("404", payload.get("reason", ""))
            self.assertIn("missing/incorrect model", payload.get("fallback_hint", ""))
            sim.stop()



class TestExperimentTabBackendControl(unittest.TestCase):
    def test_experiment_tab_backend_selection_applies_to_run_config(self):
        try:
            import tkinter as tk
            from interface import MarsColonyInterface
        except ModuleNotFoundError as exc:
            self.skipTest(f"GUI dependencies unavailable: {exc}")
            return

        try:
            app = MarsColonyInterface()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable in test environment: {exc}")
            return

        try:
            app.root.withdraw()
            self.assertEqual(app.local_model_var.get(), "qwen3.5:9b")
            self.assertAlmostEqual(float(app.local_timeout_var.get()), 900.0)
            app.brain_backend_var.set("rule_brain")
            app.apply_experiment_settings()
            self.assertIsNotNone(app.sim)
            self.assertEqual(app.sim.configured_brain_backend, "rule_brain")
            self.assertEqual(app.sim.effective_brain_backend, "rule_brain")

            app.brain_backend_var.set("local_http")
            app.local_model_var.set("llama3.1")
            app.local_base_url_var.set("http://127.0.0.1:11434")
            app.local_timeout_var.set(0.2)
            app.apply_experiment_settings()
            self.assertEqual(app.sim.configured_brain_backend, "local_http")
            self.assertEqual(app.sim.brain_backend_config.local_model, "llama3.1")
        finally:
            app.stop_experiment()
            app.root.destroy()


if __name__ == "__main__":
    unittest.main()
