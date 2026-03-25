import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from modules.agent import Agent
from modules.brain_contract import (
    AgentDIKIntegrationRequest,
    AgentDIKIntegrationResponse,
    validate_agent_dik_integration_response,
)
from modules.brain_provider import OllamaLocalBrainProvider, BrainBackendConfig, RuleBrain
from modules.knowledge import Data


class _FakeLogger:
    def __init__(self):
        self.submitted = []
        self.responses = []
        self.interpretations = []

    def record_brain_request_submitted(self, payload):
        self.submitted.append(dict(payload))

    def record_brain_response_phase(self, request_id, payload):
        self.responses.append((request_id, dict(payload)))

    def record_brain_interpretation_phase(self, request_id, payload):
        self.interpretations.append((request_id, dict(payload)))

    def log_event(self, *_args, **_kwargs):
        return None


class _FakeProvider(RuleBrain):
    def generate_dik_integration(self, request_packet):
        return AgentDIKIntegrationResponse.from_dict(
            {
                "response_id": f"resp-{request_packet.request_id}",
                "agent_id": request_packet.agent_id,
                "candidate_information_updates": [
                    {
                        "candidate_id": request_packet.candidate_information_ids[0],
                        "evidence_ids": [request_packet.held_data_ids[0]],
                        "confidence": 0.7,
                    },
                    {
                        "candidate_id": "hallucinated_info",
                        "evidence_ids": ["missing_evidence"],
                        "confidence": 0.9,
                    },
                ],
                "summary": "candidates",
                "confidence": 0.6,
            }
        )


class SplitDIKPlanningTests(unittest.TestCase):
    def _agent(self, policy="legacy"):
        return Agent(
            "Architect",
            "Architect",
            planner_config={
                "planner_request_policy": policy,
                "planning_interval_steps": 60,
                "dik_integration_cooldown_steps": 4,
                "dik_integration_batch_threshold": 1,
            },
        )

    def test_contracts_parse_and_validate(self):
        req = AgentDIKIntegrationRequest.from_dict(
            {
                "request_id": "r1",
                "tick": 2,
                "sim_time": 1.5,
                "agent_id": "a",
                "display_name": "A",
                "phase": "p",
                "trigger_reason": "new_dik_acquired",
                "held_data_ids": ["D1"],
                "held_information_ids": ["I1"],
                "held_knowledge_ids": ["K1"],
            }
        )
        self.assertEqual(req.to_dict()["held_data_ids"], ["D1"])
        resp = AgentDIKIntegrationResponse.from_dict(
            {
                "response_id": "x",
                "agent_id": "a",
                "candidate_information_updates": [{"candidate_id": "I2", "evidence_ids": ["D1"], "confidence": 0.8}],
                "confidence": 0.5,
            }
        )
        self.assertEqual(validate_agent_dik_integration_response(resp), [])

    def test_provider_normalizes_near_miss_dik_payload(self):
        provider = OllamaLocalBrainProvider(BrainBackendConfig(), RuleBrain())
        payload, steps, disposition = provider._normalize_dik_integration_payload(
            {"response": {"candidate_information_updates": [{"id": "I2", "evidence": "D1"}]}}
        )
        self.assertIsNotNone(payload)
        self.assertTrue(steps)
        self.assertEqual(disposition, "accepted_after_repair")
        self.assertEqual(payload["candidate_information_updates"][0]["candidate_id"], "I2")

    def test_split_mode_planning_is_cadence_driven(self):
        agent = self._agent(policy="cadence_with_dik_integration")
        sim = SimpleNamespace(time=10.0)
        agent.sim_step_count = 7
        allowed, reason = agent._planner_decision_allowed(sim, "new_dik_acquired")
        self.assertFalse(allowed)
        self.assertEqual(reason, "split_mode_cadence_not_due")
        agent.sim_step_count = 60
        allowed, _reason = agent._planner_decision_allowed(sim, "new_dik_acquired")
        self.assertTrue(allowed)

    def test_legacy_policy_still_allows_legacy_behavior(self):
        agent = self._agent(policy="legacy")
        sim = SimpleNamespace(time=10.0)
        agent.sim_step_count = 5
        agent.last_planner_step = -1
        allowed, _reason = agent._planner_decision_allowed(sim, "no_active_plan")
        self.assertTrue(allowed)

    def test_deterministic_acceptance_rejects_unknown_or_unsupported(self):
        agent = self._agent(policy="cadence_with_dik_integration")
        agent.mental_model["data"].add(Data("D1", "d"))
        req = AgentDIKIntegrationRequest.from_dict(
            {
                "request_id": "req",
                "tick": 1,
                "sim_time": 1.0,
                "agent_id": "Architect",
                "display_name": "Architect",
                "phase": "p",
                "trigger_reason": "new_dik_acquired",
                "held_data_ids": ["D1"],
                "held_information_ids": [],
                "held_knowledge_ids": [],
                "candidate_information_ids": ["I1"],
                "candidate_knowledge_ids": [],
                "candidate_rule_ids": [],
            }
        )
        resp = AgentDIKIntegrationResponse.from_dict(
            {
                "response_id": "r",
                "agent_id": "Architect",
                "candidate_information_updates": [
                    {"candidate_id": "I1", "evidence_ids": ["D1"], "confidence": 0.7},
                    {"candidate_id": "I999", "evidence_ids": ["D404"], "confidence": 0.8},
                ],
            }
        )
        accepted, rejected = agent._accept_dik_integration_candidates(None, req, resp)
        self.assertEqual(len(accepted["information"]), 1)
        self.assertEqual(len(rejected["information"]), 1)

    def test_planner_request_includes_accepted_integrations_only(self):
        agent = self._agent(policy="cadence_with_dik_integration")
        agent.dik_integration_state["accepted_updates"] = {
            "information": [{"candidate_id": "I2"}],
            "knowledge": [{"candidate_id": "K2"}],
            "rules": [{"candidate_id": "R2"}],
        }
        context = SimpleNamespace(
            world_snapshot={"phase_profile": {"name": "p"}},
            history_bands={"near_preceding_events": []},
            individual_cognitive_state={
                "build_readiness": {"status": "x"},
                "data_summary": ["D1"],
                "information_summary": ["I1"],
                "knowledge_summary": ["K1"],
                "known_gaps": [],
                "goal_stack": [],
                "active_plan": {},
            },
            team_state={"recent_communications": [], "externalized_artifacts": []},
            action_affordances=[],
            static_task_context={},
        )
        sim = SimpleNamespace(time=1.0, task_model=SimpleNamespace(task_id="mars"), bootstrap_reuse_enabled=False)
        req = agent._build_brain_request(sim, context, False, "test")
        self.assertEqual(req.working_memory_summary["accepted_llm_information_updates"], ["I2"])

    def test_dik_integration_observability_request_kind(self):
        agent = self._agent(policy="cadence_with_dik_integration")
        agent.mental_model["data"].add(Data("D1", "d"))
        agent.task_model = SimpleNamespace(
            dik_elements={"I1": SimpleNamespace(element_type="information", enabled=True)},
            rules={},
        )
        logger = _FakeLogger()
        sim = SimpleNamespace(
            time=1.0,
            environment=SimpleNamespace(get_current_phase=lambda: {"name": "phase"}),
            team_knowledge_manager=SimpleNamespace(recent_updates=[{"id": "m1"}], artifacts={}),
            planner_executor=ThreadPoolExecutor(max_workers=1),
            logger=logger,
            brain_provider=_FakeProvider(),
            brain_backend_config=SimpleNamespace(backend="rule_brain"),
            configured_brain_backend="rule_brain",
            effective_brain_backend="rule_brain",
            get_agent_brain_runtime=lambda _a: {"provider": _FakeProvider(), "config": SimpleNamespace(local_model="rule"), "configured_backend": "rule_brain", "effective_backend": "rule_brain"},
        )
        try:
            agent._submit_dik_integration_request_async(sim, "new_dik_acquired")
            agent._poll_dik_integration_request(sim)
        finally:
            sim.planner_executor.shutdown(wait=False)
        self.assertTrue(logger.submitted)
        self.assertEqual(logger.submitted[0]["request_kind"], "dik_integration")
        self.assertTrue(logger.interpretations)

    def test_dik_request_contains_candidate_grounding_maps(self):
        agent = self._agent(policy="cadence_with_dik_integration")
        agent.task_model = SimpleNamespace(
            dik_elements={
                "I_SYN": SimpleNamespace(element_type="information", enabled=True),
                "K_SYN": SimpleNamespace(element_type="knowledge", enabled=True),
            },
            derivations={
                "DER_I": SimpleNamespace(
                    derivation_id="DER_I",
                    output_element_id="I_SYN",
                    required_inputs=["D_A"],
                    optional_inputs=[],
                    min_required_count=1,
                    derivation_kind="deterministic",
                    enabled=True,
                ),
                "DER_K": SimpleNamespace(
                    derivation_id="DER_K",
                    output_element_id="K_SYN",
                    required_inputs=["I_SYN"],
                    optional_inputs=[],
                    min_required_count=1,
                    derivation_kind="deterministic",
                    enabled=True,
                ),
            },
            rules={
                "R_SYN": SimpleNamespace(enabled=True, required_information=["I_SYN"], required_knowledge=["K_SYN"]),
            },
        )
        sim = SimpleNamespace(
            time=3.0,
            environment=SimpleNamespace(get_current_phase=lambda: {"name": "phase"}),
            team_knowledge_manager=SimpleNamespace(recent_updates=[], artifacts={}),
        )
        request = agent._build_dik_integration_request(sim, "new_dik_acquired")
        self.assertIn("I_SYN", request.candidate_information_grounding)
        self.assertIn("K_SYN", request.candidate_knowledge_grounding)
        self.assertEqual(request.candidate_rule_grounding["R_SYN"]["required_evidence_ids"], ["I_SYN", "K_SYN"])

    def test_rule_brain_proposes_grounded_candidates_only(self):
        provider = RuleBrain()
        request = AgentDIKIntegrationRequest.from_dict(
            {
                "request_id": "dik-rb",
                "tick": 1,
                "sim_time": 1.0,
                "agent_id": "A",
                "display_name": "A",
                "phase": "p",
                "trigger_reason": "new_dik_acquired",
                "held_data_ids": ["D1"],
                "held_information_ids": [],
                "held_knowledge_ids": [],
                "candidate_information_ids": ["I1"],
                "candidate_knowledge_ids": [],
                "candidate_rule_ids": [],
                "candidate_information_grounding": {
                    "I1": [{"derivation_id": "DER1", "required_inputs": ["D1"]}],
                },
            }
        )
        response = provider.generate_dik_integration(request)
        self.assertEqual([u.candidate_id for u in response.candidate_information_updates], ["I1"])
        self.assertEqual(response.candidate_information_updates[0].evidence_ids, ["D1"])

    def test_accepted_candidates_project_into_authoritative_state(self):
        agent = self._agent(policy="cadence_with_dik_integration")
        agent.task_model = SimpleNamespace(
            dik_elements={"I2": SimpleNamespace(element_type="information", element_id="I2", description="info", role_scope="all", phase_scope="all")},
            rules={"R2": SimpleNamespace(enabled=True)},
        )
        agent.mental_model["data"].add(Data("D1", "d"))
        projected = agent._project_accepted_dik_updates(
            sim_state=SimpleNamespace(time=2.0),
            accepted_updates={
                "information": [{"candidate_id": "I2", "evidence_ids": ["D1"]}],
                "knowledge": [],
                "rules": [{"candidate_id": "R2", "evidence_ids": ["D1"]}],
            },
        )
        info_ids = {i.id for i in agent.mental_model["information"]}
        self.assertIn("I2", info_ids)
        self.assertIn("R2", agent.mental_model["knowledge"].rules)
        self.assertEqual(len(projected["information"]), 1)

    def test_update_calls_task_derivations_once_per_tick(self):
        agent = self._agent(policy="cadence_with_dik_integration")
        calls = {"count": 0}

        def _count(*_args, **_kwargs):
            calls["count"] += 1

        agent._apply_task_derivations = _count
        environment = SimpleNamespace(knowledge_packets={})
        agent.update_knowledge(environment=environment, full_packet_sweep=False, sim_state=None)
        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
