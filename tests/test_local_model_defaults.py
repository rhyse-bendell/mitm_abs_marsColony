import json
import unittest
from pathlib import Path

from interface import MarsColonyInterface


class TestLocalModelDefaults(unittest.TestCase):
    def test_default_visible_local_model_is_qwen25_3b_instruct(self):
        self.assertEqual(MarsColonyInterface.BACKEND_DEFAULTS["brain_backend"], "ollama")
        self.assertEqual(MarsColonyInterface.BACKEND_DEFAULTS["local_model"], "qwen2.5:3b-instruct")

    def test_local_model_shortlist_includes_supported_models(self):
        expected = [
            "qwen2.5:3b-instruct",
            "qwen2.5:7b",
            "qwen2.5:7b-instruct",
            "llama3.2",
            "qwen2.5:3b",
            "gemma3:4b",
            "mistral-small3.1",
            "qwen3.5:9b",
        ]
        self.assertEqual(MarsColonyInterface.LOCAL_MODEL_SHORTLIST, expected)

    def test_qwen35_9b_default_budgets_match_local_diagnostic_profile(self):
        self.assertEqual(MarsColonyInterface.BACKEND_DEFAULTS["timeout_s"], 240.0)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["backend_timeout_s"], 240.0)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["planner_timeout_seconds"], 240.0)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["warmup_timeout_seconds"], 180.0)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["startup_llm_sanity_timeout_seconds"], 240.0)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["startup_llm_sanity_completion_max_tokens"], 2048)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["planner_completion_max_tokens"], 4096)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["backend_max_retries"], 1)
        self.assertEqual(MarsColonyInterface.PLANNER_DEFAULTS["planner_max_retries"], 1)

    def test_manifest_defaults_align_with_gui_defaults(self):
        manifest_path = Path("config/tasks/mars_colony/task_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        planner = manifest["planner_defaults"]

        self.assertEqual(planner["planner_timeout_seconds"], MarsColonyInterface.PLANNER_DEFAULTS["planner_timeout_seconds"])
        self.assertEqual(planner["startup_llm_sanity_timeout_seconds"], MarsColonyInterface.PLANNER_DEFAULTS["startup_llm_sanity_timeout_seconds"])
        self.assertEqual(planner["warmup_timeout_seconds"], MarsColonyInterface.PLANNER_DEFAULTS["warmup_timeout_seconds"])
        self.assertEqual(planner["startup_llm_sanity_completion_max_tokens"], MarsColonyInterface.PLANNER_DEFAULTS["startup_llm_sanity_completion_max_tokens"])
        self.assertEqual(planner["planner_completion_max_tokens"], MarsColonyInterface.PLANNER_DEFAULTS["planner_completion_max_tokens"])


if __name__ == "__main__":
    unittest.main()
