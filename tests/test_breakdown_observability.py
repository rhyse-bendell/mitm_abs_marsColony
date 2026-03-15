import unittest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class BreakdownObservabilityTests(unittest.TestCase):
    def test_translation_failure_logs_explicit_category(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        sim.environment.knowledge_packets = {}
        decision = BrainDecision(selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="DOES_NOT_EXIST")
        agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        events = sim.logger.get_recent_events(50)
        self.assertTrue(any(e["event_type"] == "action_translation_failed" and e["payload_data"].get("failure_category") == "unresolved_target" for e in events))
        sim.stop()

    def test_target_resolution_failure_logs_category(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        sim.environment.knowledge_packets = {}
        decision = BrainDecision(selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="bad_target")
        agent._resolve_inspect_target(decision, sim.environment, sim_state=sim)
        events = sim.logger.get_recent_events(30)
        self.assertTrue(any(e["event_type"] == "target_resolution_failed" and e["payload_data"].get("failure_category") == "no_information_source_available" for e in events))
        sim.stop()

    def test_repeated_action_loop_event_and_summary_metrics_present(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        for _ in range(5):
            agent.current_plan = None
            sim.update(0.1)
        run_summary = sim.metrics.finalize()
        self.assertGreater(run_summary["events"].get("planner_invocation_requested", 0), 0)
        self.assertIn("breakdown_metrics", run_summary)
        self.assertIn("planner_invocations_by_trigger", run_summary["breakdown_metrics"])
        sim.stop()

    def test_startup_target_resolution_failure_emits_first_stage_event(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]
        sim.environment.knowledge_packets = {}
        decision = BrainDecision(selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE, target_id="bad_target")
        agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        events = sim.logger.get_recent_events(50)
        self.assertTrue(any(e["event_type"] == "first_target_resolution_failed" for e in events))
        sim.stop()

    def test_startup_movement_blocker_emits_first_stage_event(self):
        sim = SimulationState(phases=[])
        agent = sim.agents[0]

        class AlwaysBlockedEnvironment:
            objects = {"blocked": {"type": "blocked", "corners": ((0.0, 0.0), (1.0, 1.0))}}

            @staticmethod
            def is_near_object(_point, _name, threshold=0.15):
                return True

        agent.target = (7.0, 6.4)
        agent.move_toward(agent.target, dt=1.0, environment=AlwaysBlockedEnvironment(), sim_state=sim)
        events = sim.logger.get_recent_events(80)
        self.assertTrue(any(e["event_type"] == "first_movement_blocked" for e in events))
        sim.stop()


if __name__ == "__main__":
    unittest.main()
