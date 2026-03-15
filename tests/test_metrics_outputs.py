import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from modules.agent import DIK_LOG
from modules.aggregate_measures import aggregate_run_summaries
from modules.phase_definitions import MISSION_PHASES
from modules.simulation import SimulationState


class TestMetricsOutputs(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        DIK_LOG.clear()

    def _run_sim(self, tmpdir, experiment_name="metrics_run"):
        agent_configs = [
            {
                "name": "Architect",
                "role": "Architect",
                "traits": {
                    "communication_propensity": 0.9,
                    "goal_alignment": 0.9,
                    "help_tendency": 0.8,
                    "build_speed": 1.0,
                    "rule_accuracy": 0.95,
                },
                "packet_access": ["Team_Packet", "Architect_Packet"],
            },
            {
                "name": "Engineer",
                "role": "Engineer",
                "traits": {
                    "communication_propensity": 0.9,
                    "goal_alignment": 0.7,
                    "help_tendency": 0.9,
                    "build_speed": 1.0,
                    "rule_accuracy": 0.9,
                },
                "packet_access": ["Team_Packet", "Engineer_Packet"],
            },
            {
                "name": "Botanist",
                "role": "Botanist",
                "traits": {
                    "communication_propensity": 0.8,
                    "goal_alignment": 0.8,
                    "help_tendency": 0.8,
                    "build_speed": 1.0,
                    "rule_accuracy": 0.9,
                },
                "packet_access": ["Team_Packet", "Botanist_Packet"],
            },
        ]

        sim = SimulationState(
            agent_configs=agent_configs,
            phases=MISSION_PHASES,
            experiment_name=experiment_name,
            flash_mode=True,
            project_root=tmpdir,
        )
        for _ in range(30):
            sim.update(0.5)
        sim.stop()

        outputs_root = Path(tmpdir) / "Outputs"
        session_dir = [p for p in outputs_root.iterdir() if p.is_dir()][0]
        return sim, session_dir

    def test_headless_run_produces_expected_measure_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _sim, session_dir = self._run_sim(tmpdir)
            measures = session_dir / "measures"
            self.assertTrue((measures / "run_summary.json").exists())
            self.assertTrue((measures / "phase_summary.json").exists())
            self.assertTrue((measures / "agent_summary.csv").exists())
            self.assertTrue((measures / "team_summary.json").exists())

    def test_run_metadata_includes_trait_and_condition_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _sim, session_dir = self._run_sim(tmpdir)
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            metadata = run_summary["run_metadata"]
            self.assertIn("agent_traits", metadata)
            self.assertEqual(metadata["num_agents"], 3)
            self.assertIn("brain_backend", metadata)
            self.assertIn("phase_timing", metadata)
            self.assertIn("Architect", metadata["agent_traits"])
            self.assertIn("packet_access", metadata["agent_traits"]["Architect"])

    def test_phase_summaries_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _sim, session_dir = self._run_sim(tmpdir)
            phase_summary = json.loads((session_dir / "measures" / "phase_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(phase_summary), 1)
            self.assertIn("phase_name", phase_summary[0])
            self.assertIn("start_time", phase_summary[0])
            self.assertIn("end_time", phase_summary[0])

    def test_externalization_aware_metrics_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _sim, session_dir = self._run_sim(tmpdir)
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            external = run_summary["externalization_metrics"]
            self.assertIn("externalized_artifacts_created_by_type", external)
            self.assertIn("construction_artifacts_created_by_type", external)
            self.assertIn("artifact_validation_rate", external)
            self.assertIn("artifact_revision_or_repair_rate", external)

    def test_agent_and_team_summaries_have_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _sim, session_dir = self._run_sim(tmpdir)
            with (session_dir / "measures" / "agent_summary.csv").open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertIn("time_moving", rows[0])
            self.assertIn("dik_information", rows[0])

            team_summary = json.loads((session_dir / "measures" / "team_summary.json").read_text(encoding="utf-8"))
            self.assertIn("team_knowledge", team_summary)
            self.assertIn("externalization_metrics", team_summary)

    def test_aggregate_comparison_utility_outputs_combined_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._run_sim(tmpdir, experiment_name="metrics_run_a")
            self._run_sim(tmpdir, experiment_name="metrics_run_b")

            result = aggregate_run_summaries(Path(tmpdir) / "Outputs")
            self.assertEqual(result["count"], 2)
            self.assertTrue(Path(result["json"]).exists())
            self.assertTrue(Path(result["csv"]).exists())

    def test_summary_contains_planner_outcome_taxonomy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _sim, session_dir = self._run_sim(tmpdir)
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            planner = run_summary["process"]["planner_responsiveness"]
            self.assertIn("requests_completed_with_llm", planner)
            self.assertIn("requests_completed_with_fallback", planner)
            self.assertIn("llm_success_count", planner)
            self.assertIn("fallback_generated_count", planner)

            team_summary = json.loads((session_dir / "measures" / "team_summary.json").read_text(encoding="utf-8"))
            self.assertIn("plan_source_distribution", team_summary["backend"])
            self.assertIn("fallback_reason_distribution", team_summary["backend"])

    def test_headless_simulation_still_runs_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            for _ in range(5):
                sim.update(0.2)
            sim.stop()
            self.assertGreater(sim.time, 0.0)


if __name__ == "__main__":
    unittest.main()
