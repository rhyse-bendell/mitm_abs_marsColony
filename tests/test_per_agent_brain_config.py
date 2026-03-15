import csv
import json
import tempfile
import unittest
from pathlib import Path

from modules.simulation import SimulationState


class TestPerAgentBrainConfig(unittest.TestCase):
    def _agent_configs(self):
        return [
            {
                "name": "Architect",
                "role": "Architect",
                "display_name": "Lead Architect",
                "alias": "Arc",
                "label": "Arc",
                "template_id": "mars_architect",
                "traits": {},
                "brain_config": {
                    "backend": "rule_brain",
                    "local_model": "qwen3.5:9b",
                    "fallback_backend": "rule_brain",
                },
                "planner_config": {"planner_interval_steps": 2, "planner_timeout_seconds": 1.2},
            },
            {
                "name": "Engineer",
                "role": "Engineer",
                "display_name": "Systems Engineer",
                "alias": "Eng",
                "label": "Eng",
                "template_id": "mars_engineer",
                "traits": {},
                "brain_config": {
                    "backend": "ollama",
                    "local_model": "llama3.2",
                    "fallback_backend": "rule_brain",
                },
                "planner_config": {"planner_interval_steps": 5, "planner_timeout_seconds": 2.5},
            },
            {
                "name": "Botanist",
                "role": "Botanist",
                "display_name": "Botany Specialist",
                "alias": "Bot",
                "label": "Bot",
                "template_id": "mars_botanist",
                "traits": {},
                "brain_config": {
                    "backend": "local_http",
                    "local_model": "mistral",
                    "fallback_backend": "rule_brain",
                },
                "planner_config": {"planner_interval_steps": 3, "planner_timeout_seconds": 3.0},
            },
        ]

    def test_per_agent_config_propagates_to_agent_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, agent_configs=self._agent_configs())
            try:
                by_role = {a.role: a for a in sim.agents}
                self.assertEqual(by_role["Architect"].display_name, "Lead Architect")
                self.assertEqual(by_role["Engineer"].agent_label, "Eng")
                self.assertEqual(by_role["Botanist"].planner_cadence.planner_interval_steps, 3)

                runtime = sim.get_agent_brain_runtime(by_role["Engineer"])
                self.assertEqual(runtime["configured_backend"], "ollama")
                self.assertEqual(runtime["config"].local_model, "llama3.2")
                self.assertEqual(runtime["config"].fallback_backend, "rule_brain")
            finally:
                sim.stop()

    def test_default_agent_rows_still_initialize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            try:
                roles = sorted([a.role for a in sim.agents])
                self.assertEqual(roles, ["Architect", "Botanist", "Engineer"])
            finally:
                sim.stop()

    def test_display_name_and_metadata_propagate_to_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, agent_configs=self._agent_configs())
            sim.update(0.2)
            sim.stop()

            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            manifest = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(any(a.get("display_name") == "Lead Architect" for a in manifest.get("active_agents", [])))
            self.assertTrue(any(a.get("configured_backend") == "ollama" for a in manifest.get("active_agents", [])))

            logs_path = next(p for p in (session_dir / "logs").glob("*.csv") if p.name != "events.csv")
            rows = list(csv.DictReader(logs_path.open("r", encoding="utf-8")))
            self.assertTrue(any(r.get("display_name") == "Lead Architect" for r in rows))
            self.assertIn("brain_backend", rows[0])
            self.assertIn("planner_timeout_seconds", rows[0])

    def test_heterogeneous_per_agent_backend_model_and_cadence_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, agent_configs=self._agent_configs())
            try:
                runtime_map = {a.role: sim.get_agent_brain_runtime(a) for a in sim.agents}
                self.assertEqual(runtime_map["Architect"]["configured_backend"], "rule_brain")
                self.assertEqual(runtime_map["Engineer"]["configured_backend"], "ollama")
                self.assertEqual(runtime_map["Botanist"]["configured_backend"], "local_http")
                self.assertEqual(runtime_map["Botanist"]["config"].local_model, "mistral")

                cadence = {a.role: a.planner_cadence.planner_interval_steps for a in sim.agents}
                self.assertEqual(cadence, {"Architect": 2, "Engineer": 5, "Botanist": 3})
            finally:
                sim.stop()

    def test_headless_simulation_runs_with_per_agent_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, agent_configs=self._agent_configs(), flash_mode=True)
            try:
                for _ in range(2):
                    sim.update(0.2)
                self.assertGreater(sim.time, 0.0)
            finally:
                sim.stop()


class TestInterfacePerAgentDefaults(unittest.TestCase):
    def test_experiment_ui_still_has_architect_engineer_botanist_rows(self):
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
            roles = sorted(app.active_roles.keys())
            self.assertEqual(roles, ["Architect", "Botanist", "Engineer"])
            self.assertIn("Architect", app.agent_identity)
            self.assertIn("Engineer", app.agent_brain_settings)
            self.assertIn("Botanist", app.agent_planner_settings)
        finally:
            app.stop_experiment()
            app.root.destroy()


if __name__ == "__main__":
    unittest.main()
