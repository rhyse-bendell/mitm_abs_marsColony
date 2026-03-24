import unittest

from interface import MarsColonyInterface


class BrainTabHelperTests(unittest.TestCase):
    def test_brain_tab_can_render_trace_entries_and_filter_by_agent(self):
        traces = [
            {"trace_id": "t1", "agent_id": "Architect", "display_name": "Architect", "sim_time": 1.0, "tick": 4},
            {"trace_id": "t2", "agent_id": "Engineer", "display_name": "Engineer", "sim_time": 2.0, "tick": 8},
        ]
        all_rows = MarsColonyInterface._brain_trace_rows_for_agent(traces, "All")
        self.assertEqual(len(all_rows), 2)
        architect_rows = MarsColonyInterface._brain_trace_rows_for_agent(traces, "Architect")
        self.assertEqual(len(architect_rows), 1)
        self.assertEqual(architect_rows[0]["trace_id"], "t1")

    def test_brain_trace_summary_contains_disposition_and_result_source(self):
        row = {
            "sim_time": 4.2,
            "tick": 11,
            "display_name": "Architect",
            "runtime_disposition": "accepted_after_repair",
            "result_source": "ollama",
            "next_action_summary": {"action_type": "inspect_information_source"},
        }
        summary = MarsColonyInterface._brain_trace_summary_line(row)
        self.assertIn("accepted_after_repair", summary)
        self.assertIn("ollama", summary)
        self.assertIn("inspect_information_source", summary)

    def test_brain_trace_detail_includes_request_raw_normalized_and_disposition(self):
        row = {
            "request_id": "req-1",
            "trace_id": "trace-1",
            "tick": 7,
            "sim_time": 1.4,
            "configured_backend": "ollama",
            "effective_backend": "ollama",
            "model": "qwen2.5:3b",
            "trigger_reason": "plan_expired",
            "agent_brain_request_payload": {"x": 1},
            "runtime_disposition": "rejected_missing_plan",
            "planner_result": "repaired",
            "result_source": "fallback_safe_policy",
            "trace_outcome_category": "llm_invalid_with_fallback",
            "fallback_used": True,
            "fallback_reason": "provider payload missing plan",
            "next_action_summary": {"action_type": "wait"},
            "provider_trace": {
                "attempts": [
                    {
                        "raw_http_response_text": "{\"response\":{\"x\":1}}",
                        "extracted_response_payload": {"response": {"x": 1}},
                        "normalized_response_payload": None,
                        "normalization_steps": [{"step": "unwrapped_payload"}],
                    }
                ]
            },
        }
        details = MarsColonyInterface._brain_trace_detail_sections(row)
        self.assertEqual(details["request_summary"]["request_id"], "req-1")
        self.assertIn("response", details["parsed_pre_normalized"])
        self.assertEqual(details["system_interpretation"]["runtime_disposition"], "rejected_missing_plan")
        self.assertTrue(details["system_interpretation"]["fallback_used"])
        self.assertTrue(details["errors_notes"]["normalization_steps"])


if __name__ == "__main__":
    unittest.main()
