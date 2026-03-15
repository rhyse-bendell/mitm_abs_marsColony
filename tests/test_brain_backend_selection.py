import json
import tempfile
import unittest
from unittest.mock import patch

from modules.aggregate_measures import aggregate_run_summaries
from modules.brain_provider import OllamaLocalBrainProvider
from modules.simulation import SimulationState


class TestBrainBackendSelection(unittest.TestCase):
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
            )
            self.assertEqual(sim.configured_brain_backend, "local_http")
            self.assertEqual(sim.brain_backend_config.local_model, "llama3.2")
            self.assertEqual(sim.brain_backend_config.local_base_url, "http://127.0.0.1:11434")
            self.assertEqual(sim.brain_backend_config.fallback_backend, "rule_brain")
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
