import unittest

from interface import MarsColonyInterface


class BrainTabHelperTests(unittest.TestCase):
    def _row(self, request_id="req-1", agent="Architect"):
        return {
            "request_id": request_id,
            "trace_id": f"trace-{request_id}",
            "agent_id": agent,
            "display_name": agent,
            "sim_time": 1.0,
            "tick": 2,
            "request_kind": "planner",
            "request": {
                "status": "in_flight",
                "configured_backend": "ollama",
                "effective_backend": "ollama",
                "model": "qwen2.5:3b",
                "trigger_reason": "plan_expired",
                "request_payload": {"k": "v"},
            },
            "response": {
                "status": "response_received",
                "http_response_received": True,
                "json_parsed": True,
                "normalized_payload_exists": True,
                "raw_response": "{\"ok\": true}",
                "parsed_payload": {"ok": True},
                "normalized_payload": {"plan": {"next_action": {"action_type": "wait"}}},
            },
            "interpretation": {
                "status": "accepted_as_is",
                "runtime_disposition": "accepted_as_is",
                "planner_result": "accepted",
                "adopted_action": "wait",
                "fallback_used": False,
            },
        }

    def test_brain_tab_can_render_rows_and_filter_by_agent(self):
        rows = [self._row("req-1", "Architect"), self._row("req-2", "Engineer")]
        all_rows = MarsColonyInterface._brain_lifecycle_rows_for_agent(rows, "All")
        self.assertEqual(len(all_rows), 2)
        architect_rows = MarsColonyInterface._brain_lifecycle_rows_for_agent(rows, "Architect")
        self.assertEqual(len(architect_rows), 1)
        self.assertEqual(architect_rows[0]["request_id"], "req-1")

    def test_brain_rows_filtered_supports_disposition_source_and_search(self):
        rows = [self._row("req-alpha", "Architect"), self._row("req-beta", "Engineer")]
        rows[1]["interpretation"]["status"] = "timed_out"
        rows[1]["interpretation"]["runtime_disposition"] = "timed_out"
        rows[1]["request"]["effective_backend"] = "fallback_safe_policy"
        filtered = MarsColonyInterface._brain_lifecycle_rows_filtered(
            rows,
            disposition_filter="accepted_as_is",
            source_filter="ollama",
            search_text="alpha",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["request_id"], "req-alpha")

    def test_brain_visible_signature_tracks_three_pane_state(self):
        rows_a = [self._row()]
        rows_b = [self._row()]
        rows_b[0]["response"]["status"] = "normalized"
        self.assertNotEqual(MarsColonyInterface._brain_visible_signature(rows_a), MarsColonyInterface._brain_visible_signature(rows_b))

    def test_brain_detail_bundle_contains_lifecycle_sections(self):
        row = self._row()
        details = MarsColonyInterface._brain_trace_detail_sections(row)
        self.assertEqual(details["request_summary"]["request_id"], "req-1")
        self.assertIn("request_status", details["request_summary"])
        self.assertTrue(details["raw_response"])
        self.assertEqual(details["system_interpretation"]["runtime_disposition"], "accepted_as_is")

    def test_alignment_helper_uses_request_ids(self):
        rows = [self._row("req-1"), self._row("req-2")]
        self.assertEqual(MarsColonyInterface._brain_aligned_request_ids(rows), ["req-1", "req-2"])

    def test_lifecycle_bundle_for_request_is_non_empty(self):
        rows = [self._row("req-1")]
        bundle = MarsColonyInterface._brain_lifecycle_bundle_for_request(rows, "req-1")
        self.assertTrue(bundle)
        self.assertIn("request_payload", bundle)


if __name__ == "__main__":
    unittest.main()
