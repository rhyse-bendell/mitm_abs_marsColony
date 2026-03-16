import json
import tempfile
import unittest
from unittest.mock import patch

from modules.llm_sanity import SANITY_RESPONSE_FIELDS, build_startup_sanity_prompt, validate_sanity_response_schema
from modules.simulation import SimulationState


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestStartupLLMSanity(unittest.TestCase):
    def test_prompt_construction_is_bounded_and_role_grounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            agent = sim.agents[0]
            payload = build_startup_sanity_prompt(agent, sim.task_model, max_sources=1, max_items_per_type=2)
            prompt_contract = payload["prompt_contract"]

            self.assertEqual(prompt_contract["agent_identity"]["role"], agent.role)
            self.assertLessEqual(len(prompt_contract["bounded_context"]["source_ids"]), 1)
            self.assertLessEqual(len(prompt_contract["bounded_context"]["data_examples"]), 2)
            self.assertIn("raw factual units", prompt_contract["dik_framing"]["data"])
            self.assertIn("do not provide a full execution plan", prompt_contract["mission_context"]["objective"].lower())
            sim.stop()

    def test_schema_validation_accepts_valid_and_rejects_malformed(self):
        valid = {
            "agent_name": "Architect",
            "role_or_focus": "Design",
            "understood_mission": "Set up habitat safely.",
            "relevant_data_ids": ["D1"],
            "relevant_information_ids": ["I1"],
            "relevant_knowledge_or_rule_ids": ["K1"],
            "first_information_priority": "Check airlock constraints.",
            "first_coordination_need": "Sync with Engineer.",
            "confidence": 0.7,
        }
        ok, errors = validate_sanity_response_schema(valid, expected_agent_name="Architect")
        self.assertTrue(ok)
        self.assertEqual(errors, [])

        invalid = {"agent_name": "Other", "confidence": 1.9}
        ok, errors = validate_sanity_response_schema(invalid, expected_agent_name="Architect")
        self.assertFalse(ok)
        self.assertTrue(any("missing fields" in e for e in errors))
        self.assertTrue(any("confidence" in e for e in errors))

    def test_artifact_and_metadata_written_when_enabled(self):
        content = {
            field: ([] if "ids" in field else "ok") for field in SANITY_RESPONSE_FIELDS
        }
        content["agent_name"] = "Architect"
        content["confidence"] = 0.6
        llm_payload = json.dumps({
            "choices": [{"message": {"content": json.dumps(content)}}]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen", return_value=_FakeHTTPResponse(llm_payload)):
                sim = SimulationState(
                    phases=[],
                    project_root=tmpdir,
                    brain_backend="ollama",
                    planner_config={
                        "enable_startup_llm_sanity": True,
                        "startup_llm_sanity_timeout_seconds": 0.2,
                        "startup_llm_sanity_max_sources": 1,
                        "startup_llm_sanity_max_items_per_type": 1,
                    },
                )
            sim.stop()

            manifest = json.loads((sim.logger.output_session.session_folder / "session_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["startup_llm_sanity_enabled"])
            self.assertGreaterEqual(manifest["startup_llm_sanity_agent_count"], 1)
            self.assertIn("startup_llm_sanity_success_count", manifest)

            artifact_rel = manifest["startup_llm_sanity_artifact"]
            artifact = sim.logger.output_session.session_folder / artifact_rel
            self.assertTrue(artifact.exists())
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertIn("results", payload)
            self.assertGreaterEqual(len(payload["results"]), 1)

    def test_disabled_mode_keeps_startup_behavior_non_regressive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                planner_config={"enable_startup_llm_sanity": False},
            )
            sim.update(0.2)
            sim.stop()
            self.assertFalse(sim.startup_llm_sanity_summary["startup_llm_sanity_enabled"])
            self.assertIsNone(sim.startup_llm_sanity_summary["startup_llm_sanity_artifact"])

    def test_mocked_ollama_success_path_marks_validation_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                def _side_effect(req, timeout):
                    req_payload = json.loads(req.data.decode("utf-8"))
                    prompt_payload = json.loads(req_payload["messages"][1]["content"])
                    target_name = prompt_payload["agent_identity"]["agent_name"]
                    response_payload = {
                        "agent_name": target_name,
                        "role_or_focus": "role sanity",
                        "understood_mission": "Support mission startup checks.",
                        "relevant_data_ids": ["D001"],
                        "relevant_information_ids": ["I001"],
                        "relevant_knowledge_or_rule_ids": ["K001"],
                        "first_information_priority": "Verify first bounded source.",
                        "first_coordination_need": "Notify team lead.",
                        "confidence": 0.8,
                    }
                    body = json.dumps({"choices": [{"message": {"content": json.dumps(response_payload)}}]})
                    return _FakeHTTPResponse(body)

                mocked.side_effect = _side_effect
                sim = SimulationState(
                    phases=[],
                    project_root=tmpdir,
                    brain_backend="ollama",
                    planner_config={"enable_startup_llm_sanity": True},
                )
            self.assertEqual(sim.startup_llm_sanity_summary["startup_llm_sanity_failure_count"], 0)
            self.assertEqual(sim.startup_llm_sanity_summary["startup_llm_sanity_success_count"], len(sim.agents))
            sim.stop()


if __name__ == "__main__":
    unittest.main()
