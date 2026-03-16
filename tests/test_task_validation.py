import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from modules.task_model import REQUIRED_TASK_FILES, load_task_model
from modules.task_validation import run_task_validation, validate_task_model


class TaskValidationTests(unittest.TestCase):
    def test_mars_validation_runs_and_emits_artifacts(self):
        temp_dir = tempfile.mkdtemp(prefix="task_validate_out_")
        try:
            report = run_task_validation("mars_colony", output_dir=temp_dir)
            self.assertTrue((Path(temp_dir) / "task_validation_report.json").exists())
            self.assertTrue((Path(temp_dir) / "task_validation_report.md").exists())
            self.assertTrue((Path(temp_dir) / "dik_derivation_edges.csv").exists())
            self.assertIsInstance(report.unreachable_dik, list)
            self.assertIsInstance(report.unreachable_rules, list)
        finally:
            shutil.rmtree(temp_dir)

    def test_referential_integrity_detects_missing_source(self):
        temp_root = Path(tempfile.mkdtemp(prefix="task_validate_missing_source_"))
        try:
            task_dir = temp_root / "broken_task"
            task_dir.mkdir(parents=True, exist_ok=True)
            src = Path("config/tasks/mars_colony")
            for _, fname in REQUIRED_TASK_FILES.items():
                shutil.copy(src / fname, task_dir / fname)

            rows = []
            with (task_dir / "source_contents.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            rows[0]["source_id"] = "SRC_DOES_NOT_EXIST"
            with (task_dir / "source_contents.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            report = run_task_validation("broken_task", config_root=temp_root)
            codes = {issue.code for issue in report.issues}
            self.assertIn("MISSING_SOURCE", codes)
            self.assertFalse(report.passed)
        finally:
            shutil.rmtree(temp_root)

    def test_dik_reachability_detects_unreachable_element(self):
        model = load_task_model("mars_colony")
        report = validate_task_model(model)
        self.assertTrue(len(report.reachable_team) > 0)
        self.assertTrue(set(report.unreachable_dik).issubset(set(model.dik_elements.keys())))

    def test_unreachable_rule_detection_with_broken_derivation(self):
        temp_root = Path(tempfile.mkdtemp(prefix="task_validate_rule_"))
        try:
            task_dir = temp_root / "broken_rule_task"
            task_dir.mkdir(parents=True, exist_ok=True)
            src = Path("config/tasks/mars_colony")
            for _, fname in REQUIRED_TASK_FILES.items():
                shutil.copy(src / fname, task_dir / fname)

            rows = []
            with (task_dir / "dik_derivations.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            rows = [r for r in rows if r["output_element_id"] != "K_PHASE1_SUPPORT_TARGET"]
            with (task_dir / "dik_derivations.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            report = run_task_validation("broken_rule_task", config_root=temp_root)
            self.assertIn("R_PHASE1_TARGET", report.unreachable_rules)
            self.assertFalse(report.passed)
        finally:
            shutil.rmtree(temp_root)

    def test_team_access_bottleneck_detection(self):
        model = load_task_model("mars_colony")
        report = validate_task_model(model)
        self.assertIsInstance(report.team_only_rules, list)

    def test_goal_and_plan_method_grounding_reported(self):
        model = load_task_model("mars_colony")
        report = validate_task_model(model)
        self.assertIsInstance(report.unsatisfied_goals, list)
        self.assertIsInstance(report.unsatisfied_plan_methods, list)


if __name__ == "__main__":
    unittest.main()
