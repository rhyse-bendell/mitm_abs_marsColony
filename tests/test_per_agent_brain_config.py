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
                    "timeout_s": 0.8,
                    "max_retries": 1,
                },
                "planner_config": {
                    "planner_interval_steps": 2,
                    "planner_timeout_seconds": 1.2,
                    "planner_max_retries": 1,
                    "degraded_consecutive_failures_threshold": 2,
                    "degraded_cooldown_seconds": 5.0,
                    "degraded_step_interval_multiplier": 2.5,
                },
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
                    "timeout_s": 1.1,
                    "max_retries": 2,
                },
                "planner_config": {
                    "planner_interval_steps": 5,
                    "planner_timeout_seconds": 2.5,
                    "planner_max_retries": 2,
                    "degraded_consecutive_failures_threshold": 4,
                    "degraded_cooldown_seconds": 8.0,
                    "degraded_step_interval_multiplier": 3.0,
                },
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
                    "timeout_s": 2.2,
                    "max_retries": 0,
                },
                "planner_config": {
                    "planner_interval_steps": 3,
                    "planner_timeout_seconds": 3.0,
                    "planner_max_retries": 0,
                    "degraded_consecutive_failures_threshold": 3,
                    "degraded_cooldown_seconds": 6.0,
                    "degraded_step_interval_multiplier": 2.0,
                },
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
                self.assertEqual(runtime["config"].timeout_s, 1.1)
                self.assertEqual(runtime["config"].max_retries, 2)
                self.assertEqual(by_role["Architect"].planner_cadence.planner_max_retries, 1)
                self.assertEqual(by_role["Architect"].planner_cadence.degraded_consecutive_failures_threshold, 2)
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
            self.assertTrue(any(a.get("planner_max_retries") == 2 for a in manifest.get("active_agents", [])))
            self.assertTrue(any(a.get("degraded_cooldown_seconds") == 8.0 for a in manifest.get("active_agents", [])))

            logs_path = next(p for p in (session_dir / "logs").glob("*.csv") if p.name != "events.csv")
            rows = list(csv.DictReader(logs_path.open("r", encoding="utf-8")))
            self.assertTrue(any(r.get("display_name") == "Lead Architect" for r in rows))
            self.assertIn("brain_backend", rows[0])
            self.assertIn("planner_timeout_seconds", rows[0])

            events_path = session_dir / "logs" / "events.csv"
            events = list(csv.DictReader(events_path.open("r", encoding="utf-8")))
            decision_events = [e for e in events if e.get("event_type") == "brain_decision_query"]
            if decision_events:
                payload = json.loads(decision_events[0]["payload"])
                self.assertIn("configured_brain_backend", payload)
                self.assertIn("effective_brain_backend", payload)

    def test_heterogeneous_per_agent_backend_model_and_cadence_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, agent_configs=self._agent_configs())
            try:
                runtime_map = {a.role: sim.get_agent_brain_runtime(a) for a in sim.agents}
                self.assertEqual(runtime_map["Architect"]["configured_backend"], "rule_brain")
                self.assertEqual(runtime_map["Engineer"]["configured_backend"], "ollama")
                self.assertEqual(runtime_map["Botanist"]["configured_backend"], "local_http")
                self.assertEqual(runtime_map["Botanist"]["config"].local_model, "mistral")
                self.assertEqual(runtime_map["Engineer"]["config"].timeout_s, 1.1)

                cadence = {a.role: a.planner_cadence.planner_interval_steps for a in sim.agents}
                self.assertEqual(cadence, {"Architect": 2, "Engineer": 5, "Botanist": 3})
                degraded_thresholds = {a.role: a.planner_cadence.degraded_consecutive_failures_threshold for a in sim.agents}
                self.assertEqual(degraded_thresholds, {"Architect": 2, "Engineer": 4, "Botanist": 3})
            finally:
                sim.stop()

    def test_simulation_instantiates_only_selected_agent_count(self):
        configs = self._agent_configs()[:2]
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, agent_configs=configs)
            try:
                self.assertEqual(len(sim.agents), 2)
                self.assertEqual([a.role for a in sim.agents], ["Architect", "Engineer"])
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
            roles = list(app.agent_card_order)
            self.assertEqual(roles[:3], ["Architect", "Engineer", "Botanist"])
            self.assertEqual(roles[3:], ["Agent 4", "Agent 5", "Agent 6"])
            self.assertIn("Architect", app.agent_identity)
            self.assertIn("Engineer", app.agent_brain_settings)
            self.assertIn("Botanist", app.agent_planner_settings)
        finally:
            app.stop_experiment()
            app.root.destroy()

    def test_experiment_ui_defaults_and_agent_count(self):
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
            self.assertEqual(app.brain_backend_var.get(), "ollama")
            self.assertEqual(app.local_model_var.get(), "qwen3.5:9b")
            self.assertEqual(app.local_base_url_var.get(), "http://127.0.0.1:11434")
            self.assertEqual(app.local_timeout_var.get(), 15.0)
            self.assertEqual(app.fallback_backend_var.get(), "rule_brain")
            self.assertEqual(app.num_agents_var.get(), 3)
            self.assertEqual(app.agent_planner_settings["Architect"]["planner_interval_steps"].get(), 4)
            self.assertEqual(app.agent_planner_settings["Engineer"]["planner_timeout_seconds"].get(), 15.0)
            self.assertEqual(app.agent_planner_settings["Botanist"]["degraded_cooldown_seconds"].get(), 12.0)
        finally:
            app.stop_experiment()
            app.root.destroy()

    def test_build_agent_configs_respects_selected_agent_count_and_preserves_state(self):
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
            app.agent_identity["Architect"]["display_name"].set("Alpha")
            app.agent_identity["Agent 4"]["display_name"].set("Delta")
            app.num_agents_var.set(4)
            app._update_visible_agent_cards()
            cfg4 = app.build_agent_configs()
            self.assertEqual(len(cfg4), 4)
            self.assertEqual(cfg4[0]["display_name"], "Alpha")
            self.assertEqual(cfg4[3]["display_name"], "Delta")

            app.num_agents_var.set(2)
            app._update_visible_agent_cards()
            cfg2 = app.build_agent_configs()
            self.assertEqual(len(cfg2), 2)
            self.assertEqual(cfg2[0]["display_name"], "Alpha")

            app.num_agents_var.set(4)
            app._update_visible_agent_cards()
            cfg4b = app.build_agent_configs()
            self.assertEqual(cfg4b[3]["display_name"], "Delta")
        finally:
            app.stop_experiment()
            app.root.destroy()

    def test_agent_card_labels_are_dynamic_not_hardcoded_role_titles(self):
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
            self.assertEqual(app.agent_cards["Engineer"].cget("text"), "Agent 2 Configuration")
            app.agent_identity["Engineer"]["display_name"].set("Field Specialist")
            self.assertEqual(app.agent_identity["Engineer"]["display_name"].get(), "Field Specialist")
        finally:
            app.stop_experiment()
            app.root.destroy()



if __name__ == "__main__":
    unittest.main()
