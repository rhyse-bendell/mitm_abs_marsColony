import tempfile
import unittest

from modules.logging_tools import SimulationLogger


class BrainLifecycleLoggerTests(unittest.TestCase):
    def test_lifecycle_creation_on_submission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimulationLogger(project_root=tmpdir)
            logger.record_brain_request_submitted({"request_id": "req-1", "trace_id": "trace-1", "agent_id": "a1", "sim_time": 1.0, "tick": 2, "status": "in_flight"})
            rows = logger.get_recent_brain_lifecycle(10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["request"]["status"], "in_flight")
            self.assertEqual(rows[0]["response"]["status"], "pending")
            self.assertEqual(rows[0]["interpretation"]["status"], "pending")

    def test_response_and_interpretation_capture_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimulationLogger(project_root=tmpdir)
            logger.record_brain_request_submitted({"request_id": "req-1", "trace_id": "trace-1", "sim_time": 1.0, "tick": 2})
            logger.record_brain_response_phase("req-1", {"status": "response_received", "http_response_received": True, "json_parsed": True})
            logger.record_brain_interpretation_phase("req-1", {"status": "accepted_as_is", "runtime_disposition": "accepted_as_is", "request_status": "completed"})
            row = logger.get_recent_brain_lifecycle(10)[0]
            self.assertEqual(row["response"]["status"], "response_received")
            self.assertEqual(row["interpretation"]["status"], "accepted_as_is")
            self.assertEqual(row["request"]["status"], "completed")

    def test_timeout_and_transport_error_have_explicit_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimulationLogger(project_root=tmpdir)
            logger.record_brain_request_submitted({"request_id": "req-timeout", "trace_id": "trace-timeout"})
            logger.record_brain_response_phase("req-timeout", {"status": "no_response_timeout"})
            logger.record_brain_interpretation_phase("req-timeout", {"status": "timed_out"})
            logger.record_brain_request_submitted({"request_id": "req-transport", "trace_id": "trace-transport"})
            logger.record_brain_response_phase("req-transport", {"status": "transport_error"})
            logger.record_brain_interpretation_phase("req-transport", {"status": "transport_error"})
            rows = {row["request_id"]: row for row in logger.get_recent_brain_lifecycle(10)}
            self.assertEqual(rows["req-timeout"]["response"]["status"], "no_response_timeout")
            self.assertEqual(rows["req-timeout"]["interpretation"]["status"], "timed_out")
            self.assertEqual(rows["req-transport"]["response"]["status"], "transport_error")
            self.assertEqual(rows["req-transport"]["interpretation"]["status"], "transport_error")

    def test_startup_sanity_kind_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimulationLogger(project_root=tmpdir)
            logger.record_brain_request_submitted({"request_id": "startup-1", "trace_id": "trace-startup-1", "request_kind": "startup_sanity"})
            row = logger.get_recent_brain_lifecycle(10)[0]
            self.assertEqual(row["request_kind"], "startup_sanity")


if __name__ == "__main__":
    unittest.main()
