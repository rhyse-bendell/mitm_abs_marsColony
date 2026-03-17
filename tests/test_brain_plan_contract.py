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


if __name__ == "__main__":
    unittest.main()
