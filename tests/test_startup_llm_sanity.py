import json
import tempfile
import unittest
from unittest.mock import patch

from modules.llm_sanity import (
    SANITY_RESPONSE_FIELDS,
    _extract_payload_from_wrapper,
    build_startup_sanity_prompt,
    normalize_sanity_response_payload,
    run_startup_llm_sanity_check,
    validate_sanity_response_schema,
)
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
            self.assertLessEqual(len(prompt_contract["bounded_context"]["example_data_ids"]), 2)
            self.assertIn("startup sanity probe only", prompt_contract["mission_context"]["objective"].lower())
            self.assertIn("no analysis", " ".join(prompt_contract["constraints"]).lower())
            self.assertLess(len(payload["prompt_text"]), 1200)
            compact_prompt = json.loads(payload["prompt_text"])
            self.assertIn("required_response_keys", compact_prompt)
            self.assertNotIn("prompt_contract", compact_prompt)
            sim.stop()

    def test_startup_request_config_uses_larger_configurable_budget_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            requests = []

            def _side_effect(req, timeout):
                req_payload = json.loads(req.data.decode("utf-8"))
                requests.append(req_payload)
                prompt_payload = json.loads(req_payload["messages"][1]["content"])
                target_name = prompt_payload["agent_name"]
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
                return _FakeHTTPResponse(json.dumps({"choices": [{"message": {"content": json.dumps(response_payload)}}]}))

            with patch("modules.llm_sanity.request.urlopen", side_effect=_side_effect):
                sim = SimulationState(
                    phases=[],
                    project_root=tmpdir,
                    brain_backend="ollama",
                    planner_config={
                        "enable_startup_llm_sanity": True,
                        "startup_llm_sanity_completion_max_tokens": 1400,
                    },
                )
            try:
                self.assertTrue(requests)
                startup_request = next(r for r in requests if r.get("max_tokens") == 1400)
                self.assertEqual(startup_request["response_format"], {"type": "json_object"})
                self.assertIn("Do not include analysis", startup_request["messages"][0]["content"])
                self.assertNotIn("```", startup_request["messages"][1]["content"])

                artifact = sim.logger.output_session.session_folder / sim.startup_llm_sanity_summary["startup_llm_sanity_artifact"]
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                self.assertEqual(payload["startup_sanity_request_config"]["completion_max_tokens"], 1400)
                self.assertTrue(payload["startup_sanity_request_config"]["json_only_mode_requested"])
                self.assertTrue(payload["startup_sanity_request_config"]["reasoning_suppression_requested"])
                self.assertEqual(payload["results"][0]["startup_request_config"]["completion_max_tokens"], 1400)
            finally:
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
        self.assertTrue(any("must be a non-empty string" in e for e in errors))
        self.assertTrue(any("confidence" in e for e in errors))

    def test_normalization_tolerates_common_qwen_style_quirks(self):
        payload = {
            "agent_name": "",
            "role_or_focus": 42,
            "understood_mission": "Keep habitat safe",
            "relevant_data_ids": "D1,D2",
            "relevant_information_ids": "I1\nI2",
            "relevant_knowledge_or_rule_ids": None,
            "first_information_priority": "check source",
            "first_coordination_need": "sync team",
            "confidence": "80%",
        }
        normalized = normalize_sanity_response_payload(payload, expected_agent_name="Architect", fallback_role="Architect")
        ok, errors = validate_sanity_response_schema(normalized, expected_agent_name="Architect")
        self.assertTrue(ok, msg=f"unexpected errors: {errors}")
        self.assertEqual(normalized["agent_name"], "Architect")
        self.assertEqual(normalized["relevant_data_ids"], ["D1", "D2"])

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
            self.assertIn("bootstrap_reuse_enabled", manifest)
            self.assertIn("bootstrap_reuse_agent_count", manifest)
            self.assertIn("bootstrap_reuse_included_count", manifest)

            artifact_rel = manifest["startup_llm_sanity_artifact"]
            artifact = sim.logger.output_session.session_folder / artifact_rel
            self.assertTrue(artifact.exists())
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertIn("results", payload)
            self.assertGreaterEqual(len(payload["results"]), 1)


    def test_high_latency_local_defaults_raise_startup_sanity_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                brain_backend="ollama",
            )
            self.assertGreaterEqual(sim.startup_llm_sanity_config.timeout_s, 45.0)
            self.assertTrue(sim.planner_defaults.get("high_latency_local_llm_mode"))
            sim.stop()

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
                    target_name = prompt_payload["agent_name"]
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

    def test_progress_callback_emits_expected_startup_stages(self):
        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                def _side_effect(req, timeout):
                    req_payload = json.loads(req.data.decode("utf-8"))
                    prompt_payload = json.loads(req_payload["messages"][1]["content"])
                    target_name = prompt_payload["agent_name"]
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
                    planner_config={"enable_startup_llm_sanity": False},
                )
            try:
                summary = run_startup_llm_sanity_check(
                    sim,
                    config=sim.startup_llm_sanity_config,
                    progress_callback=lambda stage, current, total, agent_name, detail: events.append(
                        (stage, current, total, agent_name, detail)
                    ),
                )
                self.assertEqual(summary["startup_llm_sanity_agent_count"], len(sim.agents))
                self.assertTrue(events)
                stages = [e[0] for e in events]
                self.assertEqual(stages[0], "startup_begin")
                self.assertEqual(stages[-1], "startup_complete")
                self.assertGreaterEqual(stages.count("agent_begin"), len(sim.agents))
                self.assertGreaterEqual(stages.count("agent_result"), len(sim.agents))
            finally:
                sim.stop()


    def test_successful_startup_stores_explicit_per_agent_bootstrap_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                def _side_effect(req, timeout):
                    req_payload = json.loads(req.data.decode("utf-8"))
                    prompt_payload = json.loads(req_payload["messages"][1]["content"])
                    target_name = prompt_payload["agent_name"]
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
                    planner_config={"enable_startup_llm_sanity": True, "enable_bootstrap_summary_reuse": True},
                )
            try:
                for agent in sim.agents:
                    runtime = sim.get_agent_brain_runtime(agent)
                    bootstrap = runtime.get("bootstrap", {})
                    self.assertEqual(bootstrap.get("status"), "success")
                    self.assertIsInstance(bootstrap.get("latency_ms"), float)
                    self.assertIsInstance(bootstrap.get("validated_response"), dict)
                    self.assertTrue(bootstrap.get("summary_text"))
                    self.assertIsInstance(bootstrap.get("summary_structured"), dict)
            finally:
                sim.stop()

    def test_bootstrap_summary_reuse_included_in_planner_request_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as startup_mock:
                def _startup_side_effect(req, timeout):
                    req_payload = json.loads(req.data.decode("utf-8"))
                    prompt_payload = json.loads(req_payload["messages"][1]["content"])
                    target_name = prompt_payload["agent_name"]
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
                    return _FakeHTTPResponse(json.dumps({"choices": [{"message": {"content": json.dumps(response_payload)}}]}))

                startup_mock.side_effect = _startup_side_effect
                sim = SimulationState(
                    phases=[],
                    project_root=tmpdir,
                    brain_backend="ollama",
                    planner_config={
                        "enable_startup_llm_sanity": True,
                        "enable_bootstrap_summary_reuse": True,
                        "bootstrap_summary_max_chars": 120,
                    },
                )
            try:
                agent = sim.agents[0]
                with patch("modules.agent.uuid.uuid4") as uuid_mock:
                    uuid_mock.return_value.hex = "0123456789abcdef"
                    context = sim.brain_context_builder.build(sim, agent)
                    req = agent._build_brain_request(sim, context, request_explanation=False, trigger_reason="test")
                self.assertIsNotNone(req.bootstrap_summary)
                self.assertLessEqual(len(req.bootstrap_summary.get("summary_text", "")), 120)
                self.assertEqual(sim.startup_llm_sanity_summary["bootstrap_reuse_included_count"], 1)
            finally:
                sim.stop()

    def test_bootstrap_reuse_disabled_or_failed_preserves_planner_request_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(
                phases=[],
                project_root=tmpdir,
                planner_config={"enable_startup_llm_sanity": False, "enable_bootstrap_summary_reuse": False},
            )
            try:
                agent = sim.agents[0]
                context = sim.brain_context_builder.build(sim, agent)
                req = agent._build_brain_request(sim, context, request_explanation=False, trigger_reason="test")
                self.assertIsNone(req.bootstrap_summary)
                self.assertEqual(sim.startup_llm_sanity_summary["bootstrap_reuse_included_count"], 0)
            finally:
                sim.stop()

    def test_wrapper_with_content_blocks_extracts_and_validates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                def _side_effect(req, timeout):
                    req_payload = json.loads(req.data.decode("utf-8"))
                    prompt_payload = json.loads(req_payload["messages"][1]["content"])
                    target_name = prompt_payload["agent_name"]
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
                    body = json.dumps({
                        "choices": [{"message": {"content": [{"type": "output_text", "text": json.dumps(response_payload)}]}}]
                    })
                    return _FakeHTTPResponse(body)

                mocked.side_effect = _side_effect
                sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="ollama", planner_config={"enable_startup_llm_sanity": True})
            try:
                self.assertEqual(sim.startup_llm_sanity_summary["startup_llm_sanity_failure_count"], 0)
            finally:
                sim.stop()

    def test_empty_content_with_parseable_json_elsewhere_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                def _side_effect(req, timeout):
                    req_payload = json.loads(req.data.decode("utf-8"))
                    prompt_payload = json.loads(req_payload["messages"][1]["content"])
                    target_name = prompt_payload["agent_name"]
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
                    body = json.dumps({
                        "choices": [{"message": {"content": "", "output_text": json.dumps(response_payload)}}]
                    })
                    return _FakeHTTPResponse(body)

                mocked.side_effect = _side_effect
                sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="ollama", planner_config={"enable_startup_llm_sanity": True})
            try:
                self.assertEqual(sim.startup_llm_sanity_summary["startup_llm_sanity_failure_count"], 0)
            finally:
                sim.stop()

    def test_reasoning_only_wrapper_has_specific_failure_category(self):
        extraction = _extract_payload_from_wrapper(
            {"choices": [{"finish_reason": "length", "message": {"content": "", "reasoning": "long chain of thought"}}]}
        )
        self.assertEqual(extraction["failure_category"], "reasoning_only_response")
        self.assertTrue(extraction["truncated"])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                body = json.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "", "reasoning": "thinking only no json"}}]
                })
                mocked.return_value = _FakeHTTPResponse(body)
                sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="ollama", planner_config={"enable_startup_llm_sanity": True})
            try:
                artifact = sim.logger.output_session.session_folder / sim.startup_llm_sanity_summary["startup_llm_sanity_artifact"]
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                self.assertTrue(all(r.get("failure_category") == "reasoning_only_response" for r in payload["results"]))
            finally:
                sim.stop()

    def test_reasoning_channel_json_is_recovered(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "reasoning": json.dumps(
                            {
                                "agent_name": "Architect",
                                "role_or_focus": "Design",
                                "understood_mission": "Build safe habitat fast.",
                                "relevant_data_ids": ["D1"],
                                "relevant_information_ids": ["I1"],
                                "relevant_knowledge_or_rule_ids": ["K1"],
                                "first_information_priority": "Check constraints",
                                "first_coordination_need": "Sync engineer",
                                "confidence": 0.7,
                            }
                        ),
                    },
                }
            ]
        }
        extraction = _extract_payload_from_wrapper(payload)
        self.assertTrue(extraction.get("json_found"))
        self.assertEqual(extraction.get("candidate_source"), "choices[0].message.reasoning(recovered_json)")
        self.assertIsInstance(extraction.get("payload"), dict)

    def test_finish_reason_length_with_truncated_json_reports_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                body = json.dumps({
                    "choices": [{"finish_reason": "length", "message": {"content": '{"agent_name":"Architect"'}}]
                })
                mocked.return_value = _FakeHTTPResponse(body)
                sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="ollama", planner_config={"enable_startup_llm_sanity": True})
            try:
                artifact = sim.logger.output_session.session_folder / sim.startup_llm_sanity_summary["startup_llm_sanity_artifact"]
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                self.assertTrue(all(r.get("failure_category") == "truncated_response" for r in payload["results"]))
                self.assertTrue(all(r.get("finish_reason") == "length" for r in payload["results"]))
                self.assertTrue(all(r.get("truncated") is True for r in payload["results"]))
            finally:
                sim.stop()

    def test_finish_reason_length_with_complete_json_still_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("modules.llm_sanity.request.urlopen") as mocked:
                def _side_effect(req, timeout):
                    req_payload = json.loads(req.data.decode("utf-8"))
                    prompt_payload = json.loads(req_payload["messages"][1]["content"])
                    target_name = prompt_payload["agent_name"]
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
                    body = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": json.dumps(response_payload)}}]})
                    return _FakeHTTPResponse(body)

                mocked.side_effect = _side_effect
                sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="ollama", planner_config={"enable_startup_llm_sanity": True})
            try:
                self.assertEqual(sim.startup_llm_sanity_summary["startup_llm_sanity_failure_count"], 0)
            finally:
                sim.stop()

    def test_unrestricted_mode_propagates_large_startup_budget_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="ollama")
            try:
                self.assertTrue(sim.planner_defaults.get("unrestricted_local_qwen_mode"))
                self.assertGreaterEqual(sim.startup_llm_sanity_config.timeout_s, 120.0)
                self.assertGreaterEqual(sim.startup_llm_sanity_config.completion_max_tokens, 768)
                self.assertGreaterEqual(sim.brain_backend_config.completion_max_tokens, 2048)
                self.assertGreaterEqual(sim.brain_backend_config.warmup_timeout_s, 90.0)
            finally:
                sim.stop()


if __name__ == "__main__":
    unittest.main()
