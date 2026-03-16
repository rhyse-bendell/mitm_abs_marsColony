import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState


class TestRuntimeWitnessAudit(unittest.TestCase):
    def test_witness_path_step_can_start_and_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            audit = sim.runtime_witness_audit
            self.assertGreater(len(audit.targets), 0)
            target_id, target = next(iter(audit.targets.items()))
            first_step = target["ordered_witness_steps"][0]
            if first_step.raw_step.startswith("source_access:"):
                source_id = first_step.raw_step.split(":", 1)[1]
                sim.logger.log_event(sim.time, "source_access_succeeded", {"agent": sim.agents[0].name, "source_id": source_id, "new_information_ids": [], "new_data_ids": []})
                self.assertEqual(target["ordered_witness_steps"][0].status, "completed")
            sim.stop()

    def test_runtime_break_is_classified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "target_resolution_failed", {"agent": sim.agents[0].name, "failure_category": "unresolved_target"})
            result = sim.runtime_witness_audit.finalize()
            categories = result["summary"]["witness_step_failures_by_category"]
            self.assertIn("target_resolution_failed", categories)
            sim.stop()

    def test_generic_failure_blocking_is_agent_scoped_when_progress_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            audit = sim.runtime_witness_audit
            if len(audit.targets) < 2:
                self.skipTest("Need at least two witness targets")
            target_items = list(audit.targets.items())[:2]
            (tid_a, _), (tid_b, _) = target_items
            payload_a = {"agent": sim.agents[0].name}
            payload_b = {"agent": sim.agents[1].name}

            audit.targets[tid_a]["agents_involved"].add(sim.agents[0].name)
            audit.targets[tid_b]["agents_involved"].add(sim.agents[1].name)

            sim.logger.log_event(sim.time, "target_resolution_failed", payload_a)
            self.assertEqual(audit.targets[tid_a]["status"], "failed")
            self.assertNotEqual(audit.targets[tid_b]["status"], "failed")
            sim.stop()

    def test_runtime_witness_artifact_emitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.update(0.2)
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            artifact = session_dir / "measures" / "runtime_witness_coverage.json"
            self.assertTrue(artifact.exists())
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertIn("critical_targets", payload)

    def test_summary_metrics_include_witness_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            for _ in range(3):
                sim.update(0.2)
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            coverage = run_summary["process"]["runtime_witness_coverage"]
            self.assertIn("critical_witness_targets_total", coverage)
            self.assertIn("top_witness_failure_categories", coverage)

            team_summary = json.loads((session_dir / "measures" / "team_summary.json").read_text(encoding="utf-8"))
            self.assertIn("runtime_witness_coverage", team_summary)

    def test_mars_critical_target_is_surfaced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            payload = json.loads((session_dir / "measures" / "runtime_witness_coverage.json").read_text(encoding="utf-8"))
            target_ids = {row["target_id"] for row in payload["critical_targets"]}
            self.assertTrue(any(t.startswith("goal:") for t in target_ids))

    def test_source_reached_inspect_completed_emits_dik_and_readiness_recompute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            source_id = "Team_Info"
            target = sim.environment.get_interaction_target_position(source_id, from_position=agent.position)
            self.assertIsNotNone(target)
            agent.position = target

            with patch("modules.agent.random.random", return_value=0.0):
                changed = agent._inspect_source(sim.environment, source_id, sim_state=sim)
            self.assertTrue(changed)
            self.assertEqual(agent.source_inspection_state.get(source_id), "inspected")

            event_types = [e["event_type"] for e in sim.logger.get_recent_events(120)]
            self.assertIn("inspect_completed", event_types)
            self.assertIn("dik_acquired_from_inspect", event_types)
            self.assertIn("readiness_recomputed_after_inspect", event_types)
            sim.stop()

    def test_duplicate_inspect_restart_is_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Team_Info",
                reason_summary="test",
                confidence=0.8,
            )
            agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
            agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
            event_types = [e["event_type"] for e in sim.logger.get_recent_events(120)]
            self.assertIn("inspect_restarted_duplicate", event_types)
            sim.stop()

    def test_runtime_witness_distinguishes_inspect_and_readiness_break_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "inspect_completion_blocked", {"agent": sim.agents[0].name, "source_id": "Team_Info", "failure_category": "source_reached_inspect_not_started"})
            result = sim.runtime_witness_audit.finalize()
            categories = result["summary"]["witness_step_failures_by_category"]
            self.assertIn("inspect_not_started", categories)
            sim.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "inspect_completion_failed", {"agent": sim.agents[0].name, "source_id": "Team_Info", "failure_category": "partial_packet_uptake"})
            result = sim.runtime_witness_audit.finalize()
            categories = result["summary"]["witness_step_failures_by_category"]
            self.assertIn("inspect_completed_dik_not_acquired", categories)
            sim.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "inspect_completion_blocked", {"agent": sim.agents[0].name, "source_id": "Team_Info", "failure_category": "readiness_not_unlocked_after_inspect_success"})
            result = sim.runtime_witness_audit.finalize()
            categories = result["summary"]["witness_step_failures_by_category"]
            self.assertIn("dik_acquired_readiness_not_unlocked", categories)
            sim.stop()

    def test_successful_inspect_no_new_dik_has_distinct_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(
                sim.time,
                "inspect_post_handoff_classified",
                {
                    "agent": sim.agents[0].name,
                    "source_id": "Team_Info",
                    "post_inspect_outcome": "inspect_success_no_new_dik",
                },
            )
            result = sim.runtime_witness_audit.finalize()
            categories = result["summary"]["witness_step_failures_by_category"]
            self.assertIn("inspect_success_no_new_dik", categories)
            sim.stop()

    def test_successful_inspect_dik_and_readiness_change_are_distinct_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            source_id = "Team_Info"
            target = sim.environment.get_interaction_target_position(source_id, from_position=agent.position)
            self.assertIsNotNone(target)
            agent.position = target

            with patch("modules.agent.random.random", return_value=0.0):
                agent._inspect_source(sim.environment, source_id, sim_state=sim)

            events = sim.logger.get_recent_events(200)
            event_types = [e["event_type"] for e in events]
            self.assertIn("inspect_success_dik_changed", event_types)
            self.assertIn("inspect_success_readiness_changed", event_types)
            sim.stop()

    def test_post_inspect_handoff_prefers_productive_action_over_duplicate_inspect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            agent.post_inspect_handoff = {
                "pending": True,
                "source_id": "Team_Info",
                "dik_changed": True,
                "readiness_changed": True,
                "blockers": [],
                "blocker_category": "none",
                "outcome": "inspect_success_ready_for_action",
                "expires_at": sim.time + 10.0,
            }
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Team_Info",
                reason_summary="test",
                confidence=0.8,
            )
            translated = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
            self.assertTrue(translated)
            event_types = [e["event_type"] for e in sim.logger.get_recent_events(200)]
            self.assertTrue(
                "post_inspect_action_selected" in event_types or "post_inspect_reinspect_selected" in event_types
            )
            sim.stop()

    def test_summary_includes_post_inspect_decomposition_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "inspect_success_no_new_dik", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(sim.time, "inspect_success_derivation_triggered", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(sim.time, "inspect_success_rule_adopted", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(sim.time, "inspect_success_readiness_changed", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(sim.time, "inspect_success_no_readiness_change", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(sim.time, "post_inspect_reinspect_selected", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(sim.time, "post_inspect_action_selected", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(
                sim.time,
                "inspect_post_handoff_classified",
                {
                    "agent": sim.agents[0].name,
                    "source_id": "Team_Info",
                    "post_inspect_outcome": "inspect_success_readiness_blocked_missing_rule",
                },
            )
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            diagnostics = run_summary["process"]["inspect_readiness_diagnostics"]
            self.assertIn("inspect_success_no_new_dik_count", diagnostics)
            self.assertIn("inspect_success_derivation_triggered_count", diagnostics)
            self.assertIn("inspect_success_rule_adopted_count", diagnostics)
            self.assertIn("inspect_success_readiness_changed_count", diagnostics)
            self.assertIn("inspect_success_no_readiness_change_count", diagnostics)
            self.assertIn("post_inspect_reinspect_count", diagnostics)
            self.assertIn("post_inspect_productive_action_count", diagnostics)
            self.assertIn("post_inspect_blocker_distribution", diagnostics)

    def test_exhausted_private_source_is_deprioritized_in_target_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            agent.allowed_packet = ["Architect_Info", "Team_Info"]
            agent.source_inspection_state["Architect_Info"] = "inspected"
            agent.source_exhaustion_state["Architect_Info"] = {"inspect_count": 2, "last_dik_changed": False, "exhausted": True}

            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Architect_Info",
                reason_summary="test",
                confidence=0.8,
            )
            source_id, _ = agent._resolve_inspect_target(decision, sim.environment, sim_state=sim)
            self.assertEqual(source_id, "Team_Info")
            event_types = [e["event_type"] for e in sim.logger.get_recent_events(200)]
            self.assertIn("source_revisit_deferred", event_types)
            self.assertIn("source_revisit_suppressed", event_types)
            sim.stop()

    def test_source_exhausted_event_emitted_after_no_new_dik_reinspect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            source_id = "Team_Info"
            target = sim.environment.get_interaction_target_position(source_id, from_position=agent.position)
            self.assertIsNotNone(target)
            agent.position = target

            with patch("modules.agent.random.random", return_value=0.0):
                agent._inspect_source(sim.environment, source_id, sim_state=sim)
                changed = agent._inspect_source(sim.environment, source_id, sim_state=sim)
            self.assertFalse(changed)
            event_types = [e["event_type"] for e in sim.logger.get_recent_events(260)]
            self.assertIn("source_already_inspected_no_new_dik", event_types)
            self.assertIn("source_exhausted_for_agent", event_types)
            sim.stop()

    def test_post_private_inspect_can_prefer_shared_source_needed_by_witness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            agent.allowed_packet = ["Architect_Info", "Team_Info"]
            followup = agent._choose_post_inspect_followup_decision(sim.environment, sim_state=sim)
            self.assertEqual(followup.selected_action, ExecutableActionType.INSPECT_INFORMATION_SOURCE)
            self.assertEqual(followup.target_id, "Team_Info")
            sim.stop()

    def test_mismatch_repair_gated_when_construction_not_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            project = next(iter(sim.environment.construction.projects.values()))
            project["in_progress"] = True
            project["required_resources"] = {"bricks": 10}
            project["delivered_resources"] = {"bricks": 0}
            agent.mental_model["knowledge"].add_rule("K_ANY", [], inferred_by_agents=[agent.name])

            with patch("modules.agent.random.random", return_value=0.0):
                agent.compare_and_repair_construction(sim.environment.construction, sim_state=sim)
            event_types = [e["event_type"] for e in sim.logger.get_recent_events(200)]
            self.assertIn("mismatch_detection_skipped_not_ready", event_types)
            self.assertNotIn("construction_mismatch_detected", event_types)
            sim.stop()

    def test_movement_between_knowledge_locations_is_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            agent.position = (6.9, 1.2)
            decision = BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id="Team_Info",
                reason_summary="test",
                confidence=0.8,
            )
            agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
            event_types = [e["event_type"] for e in sim.logger.get_recent_events(180)]
            self.assertIn("movement_between_knowledge_locations", event_types)
            self.assertIn("moving_to_shared_source", event_types)
            sim.stop()

    def test_runtime_witness_shared_source_access_failure_reduced_on_reach(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            team_target = sim.environment.get_interaction_target_position("Team_Info", from_position=agent.position)
            self.assertIsNotNone(team_target)
            agent.position = team_target
            with patch("modules.agent.random.random", return_value=0.0):
                agent._inspect_source(sim.environment, "Team_Info", sim_state=sim)
            result = sim.runtime_witness_audit.finalize()
            failures = result["summary"]["witness_step_failures_by_category"]
            self.assertLess(failures.get("source_not_accessed", 0), result["summary"].get("critical_witness_targets_total", 0))
            sim.stop()

    def test_shared_source_inspect_updates_team_knowledge_and_emits_uptake_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            target = sim.environment.get_interaction_target_position("Team_Info", from_position=agent.position)
            self.assertIsNotNone(target)
            agent.position = target
            with patch("modules.agent.random.random", return_value=0.0):
                changed = agent._inspect_source(sim.environment, "Team_Info", sim_state=sim)
            self.assertTrue(changed)
            self.assertTrue(sim.team_knowledge_manager.validated_knowledge)
            event_types = [e["event_type"] for e in sim.logger.get_recent_events(320)]
            self.assertIn("shared_source_inspect_started", event_types)
            self.assertIn("shared_source_inspect_completed", event_types)
            self.assertIn("shared_source_access_success", event_types)
            self.assertIn("shared_source_dik_acquired_team", event_types)
            sim.stop()

    def test_runtime_witness_source_access_satisfied_by_shared_source_access_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            target_id, target = next(iter(sim.runtime_witness_audit.targets.items()))
            source_steps = [s for s in target["ordered_witness_steps"] if s.raw_step.startswith("source_access:")]
            if not source_steps:
                sim.stop()
                return
            source_id = source_steps[0].raw_step.split(":", 1)[1]
            sim.logger.log_event(sim.time, "shared_source_access_success", {"agent": sim.agents[0].name, "source_id": source_id, "local_dik_changed": False, "team_dik_changed": True})
            self.assertEqual(source_steps[0].status, "completed")
            sim.stop()

    def test_shared_source_mapping_block_is_explicit_in_runtime_witness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "shared_source_access_blocked", {"agent": sim.agents[0].name, "source_id": "Team_Info", "reason": "mapping_missing"})
            result = sim.runtime_witness_audit.finalize()
            categories = result["summary"]["witness_step_failures_by_category"]
            self.assertIn("shared_source_access_blocked_by_mapping", categories)
            sim.stop()

    def test_shared_source_not_marked_exhausted_when_team_state_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            agent = sim.agents[0]
            target = sim.environment.get_interaction_target_position("Team_Info", from_position=agent.position)
            self.assertIsNotNone(target)
            agent.position = target
            with patch("modules.agent.random.random", return_value=1.0):
                changed = agent._inspect_source(sim.environment, "Team_Info", sim_state=sim)
            self.assertTrue(changed)
            self.assertFalse(agent.source_exhaustion_state["Team_Info"].get("exhausted"))
            sim.stop()

    def test_summary_includes_knowledge_transition_and_suppression_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = SimulationState(phases=[], project_root=tmpdir, flash_mode=True)
            sim.logger.log_event(sim.time, "shared_source_target_selected", {"agent": sim.agents[0].name, "source_id": "Team_Info"})
            sim.logger.log_event(sim.time, "source_access_succeeded", {"agent": sim.agents[0].name, "source_id": "Team_Info", "new_information_ids": [], "new_data_ids": []})
            sim.logger.log_event(sim.time, "source_revisit_suppressed", {"agent": sim.agents[0].name, "source_id": "Architect_Info"})
            sim.logger.log_event(sim.time, "source_exhausted_for_agent", {"agent": sim.agents[0].name, "source_id": "Architect_Info"})
            sim.logger.log_event(sim.time, "movement_between_knowledge_locations", {"agent": sim.agents[0].name})
            sim.logger.log_event(sim.time, "moving_to_externalization_site", {"agent": sim.agents[0].name})
            sim.logger.log_event(sim.time, "mismatch_detection_skipped_not_ready", {"agent": sim.agents[0].name})
            sim.logger.log_event(sim.time, "repair_trigger_suppressed_not_ready", {"agent": sim.agents[0].name})
            sim.stop()
            session_dir = next((Path(tmpdir) / "Outputs").iterdir())
            run_summary = json.loads((session_dir / "measures" / "run_summary.json").read_text(encoding="utf-8"))
            diagnostics = run_summary["process"]["inspect_readiness_diagnostics"]
            self.assertIn("shared_source_target_count", diagnostics)
            self.assertIn("shared_source_access_success_count", diagnostics)
            self.assertIn("shared_source_inspect_started_count", diagnostics)
            self.assertIn("shared_source_inspect_completed_count", diagnostics)
            self.assertIn("shared_source_access_blocked_count", diagnostics)
            self.assertIn("shared_source_dik_agent_count", diagnostics)
            self.assertIn("shared_source_dik_team_count", diagnostics)
            self.assertIn("shared_source_adoption_count", diagnostics)
            self.assertIn("shared_source_exhausted_count", diagnostics)
            self.assertIn("witness_steps_satisfied_by_shared_source_count", diagnostics)
            self.assertIn("shared_source_failure_distribution", diagnostics)
            self.assertIn("private_source_revisit_suppressed_count", diagnostics)
            self.assertIn("source_exhausted_count", diagnostics)
            self.assertIn("movement_between_knowledge_locations_count", diagnostics)
            self.assertIn("externalization_target_selection_count", diagnostics)
            self.assertIn("mismatch_detection_suppressed_not_ready_count", diagnostics)
            self.assertIn("repair_trigger_suppressed_not_ready_count", diagnostics)


if __name__ == "__main__":
    unittest.main()
