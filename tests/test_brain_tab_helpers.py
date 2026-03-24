import unittest
from types import SimpleNamespace

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
        details = MarsColonyInterface._brain_lifecycle_detail_sections(row)
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

    def test_startup_sanity_rows_are_included(self):
        rows = [self._row("startup-req")]
        rows[0]["request_kind"] = "startup_sanity"
        filtered = MarsColonyInterface._brain_lifecycle_rows_filtered(rows, search_text="startup")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["request_kind"], "startup_sanity")


class _FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeCombo:
    def __init__(self):
        self.values = []

    def configure(self, **kwargs):
        if "values" in kwargs:
            self.values = list(kwargs["values"])


class _FakeListbox:
    def __init__(self):
        self.items = []
        self.selected = ()
        self.ypos = 0.0
        self.delete_calls = 0

    def delete(self, *_args):
        self.delete_calls += 1
        self.items = []
        self.selected = ()

    def insert(self, _idx, value):
        self.items.append(value)

    def yview(self, *args):
        if args:
            if args[0] == "moveto":
                self.yview_moveto(args[1])
            elif args[0] == "scroll":
                step = int(args[1])
                self.ypos = min(1.0, max(0.0, self.ypos + step * 0.05))
        return (self.ypos, min(1.0, self.ypos + 0.25))

    def yview_moveto(self, value):
        self.ypos = float(value)

    def curselection(self):
        return self.selected

    def selection_clear(self, *_args):
        self.selected = ()

    def selection_set(self, idx):
        self.selected = (idx,)

    def activate(self, _idx):
        return None

    def size(self):
        return len(self.items)


class _FakeText:
    def __init__(self):
        self.content = ""
        self.ypos = 0.0

    def delete(self, *_args):
        self.content = ""

    def insert(self, _idx, value):
        self.content += str(value)

    def yview(self):
        return (self.ypos, min(1.0, self.ypos + 0.25))

    def yview_moveto(self, value):
        self.ypos = float(value)

    def get(self, *_args):
        return self.content


class _FakeLoggerLifecycleOnly:
    def __init__(self, rows):
        self.rows = rows
        self.lifecycle_calls = 0

    def get_recent_brain_lifecycle(self, _limit):
        self.lifecycle_calls += 1
        return list(self.rows)


class _FakeLoggerWithPlannerAlso(_FakeLoggerLifecycleOnly):
    def __init__(self, rows):
        super().__init__(rows)
        self.planner_calls = 0

    def get_recent_planner_traces(self, _limit):
        self.planner_calls += 1
        return [{"request_id": "planner-only"}]


class BrainTabWiringTests(unittest.TestCase):
    def _build_app(self, rows, *, logger_cls=_FakeLoggerWithPlannerAlso):
        app = MarsColonyInterface.__new__(MarsColonyInterface)
        app.sim = SimpleNamespace(logger=logger_cls(rows))
        app.brain_request_list = _FakeListbox()
        app.brain_response_list = _FakeListbox()
        app.brain_interpretation_list = _FakeListbox()
        app.brain_detail_text = _FakeText()
        app.brain_agent_filter = _FakeVar("All")
        app.brain_disposition_filter = _FakeVar("All")
        app.brain_source_filter = _FakeVar("All")
        app.brain_search_var = _FakeVar("")
        app.brain_auto_refresh_var = _FakeVar(True)
        app.brain_follow_latest_var = _FakeVar(True)
        app.brain_agent_combo = _FakeCombo()
        app.brain_disposition_combo = _FakeCombo()
        app.brain_source_combo = _FakeCombo()
        app._brain_visible_rows = []
        app._brain_visible_signature_cache = ()
        app._brain_last_detail_key = None
        app._brain_user_inspecting = False
        return app

    def test_update_brain_tab_prefers_lifecycle_not_planner_trace_stream(self):
        row = BrainTabHelperTests()._row("req-live")
        app = self._build_app([row])
        app.update_brain_tab(force=True)
        self.assertEqual(app.sim.logger.lifecycle_calls, 1)
        self.assertEqual(app.sim.logger.planner_calls, 0)
        self.assertEqual(MarsColonyInterface._brain_aligned_request_ids(app._brain_visible_rows), ["req-live"])

    def test_three_panes_stay_aligned_and_selection_syncs(self):
        rows = [BrainTabHelperTests()._row("req-1"), BrainTabHelperTests()._row("req-2")]
        app = self._build_app(rows)
        app.update_brain_tab(force=True)
        self.assertEqual(len(app.brain_request_list.items), 2)
        self.assertEqual(len(app.brain_response_list.items), 2)
        self.assertEqual(len(app.brain_interpretation_list.items), 2)
        app.brain_response_list.selection_set(0)
        app._on_brain_list_selection_changed("response")
        self.assertEqual(app.brain_request_list.curselection(), (0,))
        self.assertEqual(app.brain_interpretation_list.curselection(), (0,))
        app.brain_interpretation_list.selection_set(1)
        app._on_brain_list_selection_changed("interpretation")
        self.assertEqual(app.brain_request_list.curselection(), (1,))
        self.assertEqual(app.brain_response_list.curselection(), (1,))

    def test_detail_and_copy_bundle_are_lifecycle_based(self):
        row = BrainTabHelperTests()._row("req-copy")
        row["interpretation"]["failure_mode"] = "none"
        app = self._build_app([row], logger_cls=_FakeLoggerLifecycleOnly)
        copied = {"value": ""}
        app._copy_to_clipboard = lambda content, **_kwargs: copied.__setitem__("value", str(content))
        app.update_brain_tab(force=True)
        app._sync_brain_list_selection(0)
        app._update_brain_detail(force=True)
        self.assertIn("Request summary", app.brain_detail_text.get("1.0", "end-1c"))
        self.assertIn("System interpretation", app.brain_detail_text.get("1.0", "end-1c"))
        app._copy_brain_lifecycle_bundle()
        self.assertIn("request_summary", copied["value"])
        self.assertIn("system_interpretation", copied["value"])

    def test_refresh_stability_keeps_selection_when_signature_unchanged(self):
        row = BrainTabHelperTests()._row("req-stable")
        app = self._build_app([row], logger_cls=_FakeLoggerLifecycleOnly)
        app.update_brain_tab(force=True)
        app._sync_brain_list_selection(0)
        app.brain_request_list.yview_moveto(0.4)
        before_items = list(app.brain_request_list.items)
        app.update_brain_tab(force=False)
        self.assertEqual(before_items, app.brain_request_list.items)
        self.assertEqual(app.brain_request_list.curselection(), (0,))
        self.assertAlmostEqual(app.brain_request_list.yview()[0], 0.4)
        self.assertEqual(app.brain_request_list.delete_calls, 1)

    def test_summary_lines_include_three_pane_lifecycle_fields(self):
        row = BrainTabHelperTests()._row("req-rich")
        row["request_kind"] = "startup_sanity"
        row["response"]["status"] = "timeout"
        row["interpretation"]["failure_mode"] = "no_usable_plan"
        req_line = MarsColonyInterface._brain_request_summary_line(row)
        resp_line = MarsColonyInterface._brain_response_summary_line(row)
        interp_line = MarsColonyInterface._brain_interpretation_summary_line(row)
        self.assertIn("kind=startup_sanity", req_line)
        self.assertIn("model=", req_line)
        self.assertIn("timeout", resp_line)
        self.assertIn("normalized=", resp_line)
        self.assertIn("failure=no_usable_plan", interp_line)


if __name__ == "__main__":
    unittest.main()
