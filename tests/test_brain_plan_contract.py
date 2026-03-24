import json
import tempfile
import unittest
from unittest.mock import patch

from modules.brain_contract import (
    AgentBrainRequest,
    AgentBrainResponse,
    BRAIN_REQUEST_JSON_SCHEMA,
    BRAIN_RESPONSE_JSON_SCHEMA,
    validate_agent_brain_response,
)
from modules.brain_provider import BrainBackendConfig, OllamaLocalBrainProvider, RuleBrain
from modules.simulation import SimulationState


class BrainContractTests(unittest.TestCase):
    def test_request_and_response_schema_constants_exist(self):
        self.assertEqual(BRAIN_REQUEST_JSON_SCHEMA["type"], "object")
        self.assertEqual(BRAIN_RESPONSE_JSON_SCHEMA["type"], "object")

    def test_agent_plan_validation_flags_illegal_next_action(self):
        response = AgentBrainResponse.from_dict(
            {
                "response_id": "r1",
                "agent_id": "a1",
                "plan": {
                    "plan_id": "p1",
                    "plan_horizon": 2,
                    "ordered_goals": [],
                    "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "hold"}],
                    "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "hold"},
                    "confidence": 0.9,
                },
            }
        )
        errors = validate_agent_brain_response(response, ["inspect_information_source"])
        self.assertTrue(errors)

    def test_explanation_scheduling_every_n_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, planner_config={"explanation_mode": "every_n_calls", "explanation_every_n_calls": 2})
            agent = sim.agents[0]
            self.assertFalse(agent._should_request_explanation())
            agent.planner_call_count = 1
            self.assertTrue(agent._should_request_explanation())

    def test_generic_agent_template_compatibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir)
            self.assertTrue(all(getattr(a, "agent_id", None) for a in sim.agents))
            self.assertTrue(all(getattr(a, "display_name", None) for a in sim.agents))

    def test_mock_provider_round_trip_with_plan_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, brain_backend="local_stub")
            for _ in range(4):
                sim.update(0.2)
            self.assertIsNotNone(sim.agents[0].current_plan)


    def test_provider_request_payload_includes_configured_completion_budget(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", completion_max_tokens=7777, max_retries=0), fallback=RuleBrain())
        req = AgentBrainRequest(
            request_id="q2",
            tick=1,
            sim_time=1.0,
            agent_id="a1",
            display_name="A1",
            agent_label="A",
            task_id="mars_colony",
            phase="p1",
            local_context_summary="ctx",
            local_observations=[],
            working_memory_summary={},
            inbox_summary=[],
            current_goal_stack=[],
            current_plan_summary={},
            allowed_actions=[{"action_type": "wait"}],
            planning_horizon_config={"max_steps": 2},
            request_explanation=False,
        )

        class _FakeResponse:
            def __init__(self, payload):
                self.payload = payload
            def read(self):
                return self.payload.encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False

        captured = {}
        def _urlopen(req_obj, timeout):
            captured["payload"] = req_obj.data.decode("utf-8")
            body = '{"choices":[{"message":{"content":"{\"response_id\":\"r\",\"agent_id\":\"a1\",\"plan\":{\"plan_id\":\"p\",\"plan_horizon\":1,\"ordered_goals\":[],\"ordered_actions\":[{\"step_index\":0,\"action_type\":\"wait\",\"expected_purpose\":\"ok\"}],\"next_action\":{\"step_index\":0,\"action_type\":\"wait\",\"expected_purpose\":\"ok\"},\"confidence\":0.7}}"}}]}'
            return _FakeResponse(body)

        with patch("modules.brain_provider.request.urlopen", side_effect=_urlopen):
            provider.generate_plan(req)
        self.assertIn('"max_tokens": 7777', captured["payload"])

    def test_provider_request_payload_is_compact_and_bounded(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", completion_max_tokens=7777, max_retries=0), fallback=RuleBrain())
        req = AgentBrainRequest(
            request_id="q3",
            tick=1,
            sim_time=1.0,
            agent_id="a1",
            display_name="A1",
            agent_label="A",
            task_id="mars_colony",
            phase="p1",
            local_context_summary="ctx " * 120,
            local_observations=[f"obs-{i}" for i in range(20)],
            working_memory_summary={"huge": "x" * 6000},
            inbox_summary=[],
            current_goal_stack=[{"goal_id": f"g{i}", "notes": "x" * 1000} for i in range(12)],
            current_plan_summary={"notes": "x" * 5000},
            allowed_actions=[{"action_type": "wait", "meta": "x" * 1000, "idx": i} for i in range(30)],
            planning_horizon_config={"max_steps": 2},
            request_explanation=False,
            task_context={"very_large": "x" * 10000},
            rule_context=["r" * 1000 for _ in range(20)],
            derivation_context=["d" * 1000 for _ in range(20)],
            artifact_context=[{"artifact_id": f"a{i}", "body": "x" * 1000} for i in range(20)],
        )
        payload = provider._build_request_payload(req)
        user_contract = json.loads(payload["messages"][1]["content"])
        self.assertNotIn("task_context", user_contract)
        self.assertLessEqual(len(user_contract["allowed_actions"]), 14)
        self.assertLessEqual(len(user_contract["local_observations"]), 6)
        self.assertLessEqual(len(user_contract["current_goal_stack"]), 5)
        self.assertLess(len(payload["messages"][1]["content"]), len(json.dumps(req.to_dict(), default=str)))

    def test_parse_response_recovers_json_from_reasoning_when_content_empty(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        wrapped = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": json.dumps(
                            {
                                "response_id": "r",
                                "agent_id": "a1",
                                "plan": {
                                    "plan_id": "p",
                                    "plan_horizon": 1,
                                    "ordered_goals": [],
                                    "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "ok"}],
                                    "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "ok"},
                                    "confidence": 0.8,
                                },
                            }
                        ),
                    }
                }
            ]
        }
        result = provider._parse_response(wrapped)
        self.assertEqual(result["parse_source"], "choices[0].message.reasoning")
        self.assertEqual(result["payload"]["agent_id"], "a1")

    def test_runtime_payload_enforces_json_only_contract(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        req = AgentBrainRequest(
            request_id="q4",
            tick=1,
            sim_time=1.0,
            agent_id="a1",
            display_name="A1",
            agent_label="A",
            task_id="mars_colony",
            phase="p1",
            local_context_summary="ctx",
            local_observations=[],
            working_memory_summary={},
            inbox_summary=[],
            current_goal_stack=[],
            current_plan_summary={},
            allowed_actions=[{"action_type": "wait"}],
            planning_horizon_config={"max_steps": 2},
            request_explanation=False,
        )
        payload = provider._build_request_payload(req)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("Return exactly one JSON object", payload["messages"][0]["content"])
        self.assertIn("Do not wrap the object under keys like response/result/data", payload["messages"][0]["content"])

    def test_normalize_payload_accepts_top_level_plan(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        payload = {
            "response_id": "r",
            "agent_id": "a1",
            "plan": {
                "plan_id": "p",
                "plan_horizon": 1,
                "ordered_goals": [],
                "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "ok"}],
                "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "ok"},
                "confidence": 0.8,
            },
        }
        normalized, steps, disposition = provider._normalize_payload(payload)
        self.assertIsNotNone(normalized)
        self.assertEqual(disposition, "accepted_as_is")
        self.assertFalse(steps)

    def test_normalize_payload_unwraps_response_plan(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        wrapped = {
            "response": {
                "response_id": "r",
                "agent_id": "a1",
                "plan": {
                    "plan_id": "p",
                    "plan_horizon": 1,
                    "ordered_goals": [],
                    "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "ok"}],
                    "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "ok"},
                    "confidence": 0.8,
                },
            }
        }
        normalized, steps, disposition = provider._normalize_payload(wrapped)
        self.assertIsNotNone(normalized)
        self.assertEqual(disposition, "accepted_after_repair")
        self.assertTrue(any(step.get("step") == "unwrapped_payload" for step in steps))

    def test_normalize_payload_unwraps_result_plan(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        wrapped = {
            "result": {
                "response_id": "r",
                "agent_id": "a1",
                "plan": {
                    "plan_id": "p",
                    "plan_horizon": 1,
                    "ordered_goals": [],
                    "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "ok"}],
                    "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "ok"},
                    "confidence": 0.8,
                },
            }
        }
        normalized, _steps, disposition = provider._normalize_payload(wrapped)
        self.assertIsNotNone(normalized)
        self.assertEqual(disposition, "accepted_after_repair")

    def test_normalize_payload_rejects_missing_plan(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        normalized, _steps, disposition = provider._normalize_payload({"response": {"x": 1}})
        self.assertIsNone(normalized)
        self.assertEqual(disposition, "rejected_missing_plan")

    def test_normalize_payload_repairs_string_next_action_from_ordered_actions(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        payload = {
            "response_id": "r",
            "agent_id": "a1",
            "plan": {
                "plan_id": "p",
                "plan_horizon": 1,
                "ordered_goals": [],
                "ordered_actions": [{"step_index": 0, "action_type": "inspect", "expected_purpose": "inspect"}],
                "next_action": "inspect_information_source",
                "confidence": 0.8,
            },
        }
        normalized, steps, disposition = provider._normalize_payload(payload)
        self.assertEqual(disposition, "accepted_after_repair")
        self.assertEqual(normalized["plan"]["next_action"]["action_type"], "inspect_information_source")
        self.assertTrue(any(step.get("step") == "resolved_string_next_action_from_ordered_actions" for step in steps))

    def test_generate_plan_records_normalization_steps_in_trace(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        req = AgentBrainRequest(
            request_id="q5",
            tick=1,
            sim_time=1.0,
            agent_id="a1",
            display_name="A1",
            agent_label="A",
            task_id="mars_colony",
            phase="p1",
            local_context_summary="ctx",
            local_observations=[],
            working_memory_summary={},
            inbox_summary=[],
            current_goal_stack=[],
            current_plan_summary={},
            allowed_actions=[{"action_type": "wait"}, {"action_type": "inspect_information_source"}],
            planning_horizon_config={"max_steps": 2},
            request_explanation=False,
        )

        class _FakeResponse:
            def __init__(self, payload):
                self.payload = payload
            def read(self):
                return self.payload.encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False

        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "response": {
                                        "response_id": "r",
                                        "agent_id": "a1",
                                        "plan": {
                                            "plan_id": "p",
                                            "plan_horizon": 1,
                                            "ordered_goals": [],
                                            "ordered_actions": ["inspect"],
                                            "next_action": "inspect_information_source",
                                            "confidence": 0.8,
                                        },
                                    }
                                }
                            )
                        }
                    }
                ]
            }
        )
        with patch("modules.brain_provider.request.urlopen", return_value=_FakeResponse(body)):
            provider.generate_plan(req)
        attempt = provider.last_trace["attempts"][-1]
        self.assertTrue(attempt.get("normalization_steps"))
        self.assertEqual(provider.last_trace.get("runtime_disposition"), "accepted_after_repair")

class OllamaProviderFallbackTests(unittest.TestCase):
    def test_ollama_provider_falls_back_on_malformed_payload(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        req = AgentBrainRequest(
            request_id="q1",
            tick=1,
            sim_time=1.0,
            agent_id="a1",
            display_name="A1",
            agent_label="A",
            task_id="mars_colony",
            phase="p1",
            local_context_summary="ctx",
            local_observations=[],
            working_memory_summary={},
            inbox_summary=[],
            current_goal_stack=[],
            current_plan_summary={},
            allowed_actions=[{"action_type": "wait"}],
            planning_horizon_config={"max_steps": 2},
            request_explanation=False,
        )
        with patch("modules.brain_provider.request.urlopen", side_effect=TimeoutError("timeout")):
            response = provider.generate_plan(req)
        self.assertEqual(response.plan.next_action.action_type.value, "wait")

    def test_ollama_provider_rejects_missing_plan_with_explicit_disposition(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(backend="ollama", max_retries=0), fallback=RuleBrain())
        req = AgentBrainRequest(
            request_id="q6",
            tick=1,
            sim_time=1.0,
            agent_id="a1",
            display_name="A1",
            agent_label="A",
            task_id="mars_colony",
            phase="p1",
            local_context_summary="ctx",
            local_observations=[],
            working_memory_summary={},
            inbox_summary=[],
            current_goal_stack=[],
            current_plan_summary={},
            allowed_actions=[{"action_type": "wait"}],
            planning_horizon_config={"max_steps": 2},
            request_explanation=False,
        )

        class _FakeResponse:
            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"response\\":{\\"x\\":1}}"}}]}'
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False

        with patch("modules.brain_provider.request.urlopen", return_value=_FakeResponse()):
            provider.generate_plan(req)
        self.assertEqual(provider.last_trace.get("runtime_disposition"), "rejected_missing_plan")


if __name__ == "__main__":
    unittest.main()
