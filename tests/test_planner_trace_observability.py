import json
import tempfile
import time
import unittest
from unittest.mock import patch

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


def _read_trace_rows(sim):
    session_dir = next(sim.logger.output_session.outputs_root.iterdir())
    trace_path = session_dir / "logs" / "planner_trace.jsonl"
    if not trace_path.exists():
        return [], trace_path
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, trace_path


class TestPlannerTraceObservability(unittest.TestCase):
    def _drive_planner(self, sim, steps=6, dt=0.1):
        for _ in range(steps):
            sim.update(dt)
            time.sleep(0.02)

    def _build_sim(self, tmpdir, backend="local_http", planner_config=None):
        cfg = {
            "planner_interval_steps": 1,
            "planner_interval_time": 0.0,
            "planner_timeout_seconds": 0.2,
            "enable_planner_trace": True,
            "planner_trace_mode": "full",
            "planner_trace_max_chars": 8000,
        }
        if planner_config:
            cfg.update(planner_config)
        return SimulationState(
            phases=[],
            project_root=tmpdir,
            brain_backend=backend,
            brain_backend_options={"timeout_s": 0.1, "max_retries": 0, "fallback_backend": "rule_brain"},
            planner_config=cfg,
        )

    def test_successful_planner_call_persists_full_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)

            payload = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "response_id": "resp-1",
                                    "agent_id": sim.agents[0].agent_id,
                                    "plan": {
                                        "plan_id": "p-1",
                                        "plan_horizon": 1,
                                        "ordered_goals": [],
                                        "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "hold"}],
                                        "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "hold"},
                                        "confidence": 0.5,
                                    },
                                }
                            )
                        }
                    }
                ]
            }

            with patch("modules.brain_provider.request.urlopen", return_value=_FakeHTTPResponse(json.dumps(payload))):
                self._drive_planner(sim)
            sim.stop()

            rows, _ = _read_trace_rows(sim)
            self.assertTrue(rows)
            row = rows[-1]
            self.assertIn("trace_id", row)
            self.assertIn("agent_brain_request_payload", row)
            self.assertIn("raw_http_response_text", row)
            self.assertIn("parsed_response_json", row)
            self.assertIn("extracted_response_payload", row)
            self.assertIn("normalized_agent_brain_response", row)
            self.assertEqual(row.get("plan_disposition"), "adopted")
            self.assertTrue(row.get("llm_response_received"))
            self.assertTrue(row.get("llm_response_validated"))
            self.assertEqual(row.get("trace_outcome_category"), "llm_success")
            self.assertEqual(row.get("result_source"), "ollama")

    def test_timeout_or_failure_still_emits_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)

            def slow_generate(self, request_packet):
                time.sleep(0.3)
                return self.fallback.generate_plan(request_packet)

            with patch("modules.brain_provider.OllamaLocalBrainProvider.generate_plan", new=slow_generate):
                for _ in range(4):
                    sim.update(0.1)
            sim.stop()

            rows, _ = _read_trace_rows(sim)
            failed = [r for r in rows if r.get("planner_result") in {"timed_out", "failed"}]
            self.assertTrue(failed)
            self.assertTrue(any(r.get("fallback") for r in failed))
            self.assertTrue(any(r.get("timeout_occurred") for r in failed))
            self.assertTrue(any(r.get("fallback_used") for r in failed))
            self.assertTrue(any(r.get("trace_outcome_category") == "llm_timeout_with_fallback" for r in failed))

    def test_validation_failure_trace_keeps_partial_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir)
            malformed = {"choices": [{"message": {"content": "{not-json"}}]}
            with patch("modules.brain_provider.request.urlopen", return_value=_FakeHTTPResponse(json.dumps(malformed))):
                self._drive_planner(sim)
            sim.stop()

            rows, _ = _read_trace_rows(sim)
            self.assertTrue(rows)
            fallback_rows = [r for r in rows if r.get("fallback")]
            self.assertTrue(fallback_rows)
            row = fallback_rows[-1]
            self.assertIsNotNone(row.get("raw_http_response_text"))
            self.assertIn("provider_attempts", row)
            self.assertEqual(row.get("result_source"), "fallback_safe_policy")
            self.assertTrue(row.get("fallback_used"))

    def test_events_link_to_trace_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir, backend="rule_brain")
            self._drive_planner(sim)
            sim.stop()
            events = sim.logger.get_recent_events(200)
            linked = [e for e in events if e.get("event_type") in {"planner_request_started_async", "planner_invocation_completed", "brain_provider_response_received"}]
            self.assertTrue(linked)
            for event in linked:
                payload = event.get("payload_data") or json.loads(event.get("payload", "{}"))
                self.assertIn("trace_id", payload)

    def test_manifest_and_trace_artifact_integration_and_headless_compat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = self._build_sim(tmpdir, backend="rule_brain")
            self._drive_planner(sim)
            sim.stop()
            session_dir = next(sim.logger.output_session.outputs_root.iterdir())
            manifest = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest.get("planner_trace_enabled"))
            self.assertEqual(manifest.get("planner_trace_artifact"), "logs/planner_trace.jsonl")
            self.assertTrue((session_dir / "logs" / "planner_trace.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
