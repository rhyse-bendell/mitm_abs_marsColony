import json
import tempfile
import unittest
from pathlib import Path

from modules.simulation import SimulationState


class TestRuntimeWitnessAudit(unittest.TestCase):
    def test_witness_path_step_can_start_and_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            audit = sim.runtime_witness_audit
            self.assertGreater(len(audit.targets), 0)
            target_id, target = next(iter(audit.targets.items()))
            first_step = target["ordered_witness_steps"][0]
            if first_step.raw_step.startswith("source_access:"):
                source_id = first_step.raw_step.split(":", 1)[1]
                sim.logger.log_event(sim.time, "source_access_succeeded", {"agent": sim.agents[0].name, "source_id": source_id, "new_information_ids": [], "new_data_ids": []})
                self.assertEqual(target["ordered_witness_steps"][0].status, "completed")
            sim.stop()

    def test_runtime_break_is_classified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "target_resolution_failed", {"agent": sim.agents[0].name, "failure_category": "unresolved_target"})
            result = sim.runtime_witness_audit.finalize()
            categories = result["summary"]["witness_step_failures_by_category"]
            self.assertIn("target_resolution_failed", categories)
            sim.stop()

    def test_runtime_witness_artifact_emitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.update(0.2)
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            artifact = session_dir / "measures" / "runtime_witness_coverage.json"
            self.assertTrue(artifact.exists())
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertIn("critical_targets", payload)

    def test_summary_metrics_include_witness_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            for _ in range(3):
                sim.update(0.2)
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            coverage = run_summary["process"]["runtime_witness_coverage"]
            self.assertIn("critical_witness_targets_total", coverage)
            self.assertIn("top_witness_failure_categories", coverage)

            team_summary = json.loads((session_dir / "measures" / "team_summary.json").read_text(encoding="utf-8"))
            self.assertIn("runtime_witness_coverage", team_summary)

    def test_mars_critical_target_is_surfaced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            payload = json.loads((session_dir / "measures" / "runtime_witness_coverage.json").read_text(encoding="utf-8"))
            target_ids = {row["target_id"] for row in payload["critical_targets"]}
            self.assertTrue(any(t.startswith("goal:") for t in target_ids))


if __name__ == "__main__":
    unittest.main()
