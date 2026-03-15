import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from modules.agent import DIK_LOG


class MetricsCollector:
    """Runtime metrics accumulator for comparison-ready run outputs."""

    ZONE_MAP = {
        "Zone_Architect_Info": "architect_info",
        "Zone_Engineer_Info": "engineer_info",
        "Zone_Botanist_Info": "botanist_info",
        "Zone_Team_Info": "team_info",
        "Zone_Table_A": "table_a",
        "Zone_Table_B": "table_b",
        "Zone_Table_C": "table_c",
    }

    def __init__(self, simulation):
        self.simulation = simulation
        self.started_at = simulation.time
        self._last_phase_index = None
        self._phase_open_time = simulation.time

        self.events_by_type = Counter()
        self.breakdown_counts = defaultdict(Counter)
        self.reason_distributions = defaultdict(Counter)
        self.communication_by_type = Counter()
        self.dik_counts = defaultdict(Counter)
        self.externalization_by_type = Counter()
        self.construction_externalizations_by_type = Counter()
        self.construction_creation_by_type = Counter()
        self.construction_revision_by_type = Counter()
        self.construction_validation_by_type = Counter()
        self._seen_construction_projects = set()
        self._artifact_state = {}

        self.agent_stats = {
            agent.name: {
                "agent": agent.name,
                "role": agent.role,
                "traits": {
                    "communication_propensity": getattr(agent, "communication_propensity", 0.5),
                    "goal_alignment": getattr(agent, "goal_alignment", 0.5),
                    "help_tendency": getattr(agent, "help_tendency", 0.5),
                    "build_speed": getattr(agent, "build_speed", 0.5),
                    "rule_accuracy": getattr(agent, "rule_accuracy", 0.5),
                },
                "packet_access": list(getattr(agent, "allowed_packet", [])),
                "time_moving": 0.0,
                "time_stalled": 0.0,
                "time_inspecting_info": 0.0,
                "time_communicating": 0.0,
                "time_transporting": 0.0,
                "time_externalizing": 0.0,
                "time_constructing": 0.0,
                "time_validating_or_repairing": 0.0,
                "retarget_count": 0,
                "stall_episode_count": 0,
                "mismatch_detected": 0,
                "repair_episodes": 0,
                "help_requests": 0,
                "plan_externalizations": 0,
                "artifact_consultations": 0,
                "phase_transitions_seen": 0,
                "planning_or_externalization_actions": 0,
                "execution_actions": 0,
                "correction_validation_actions": 0,
                "zone_time": Counter(),
                "dik": Counter(),
                "knowledge_rules_end": 0,
                "_last_status": "",
            }
            for agent in simulation.agents
        }

        self.phase_stats = []
        self._open_phase()

    def _open_phase(self):
        phase = self.simulation.environment.get_current_phase() or {"name": "default"}
        self._last_phase_index = self.simulation.environment.current_phase_index
        self._phase_open_time = self.simulation.time
        self.phase_stats.append(
            {
                "phase_name": phase.get("name", "default"),
                "phase_index": self._last_phase_index,
                "start_time": round(self.simulation.time, 3),
                "end_time": None,
                "events": Counter(),
                "structures_completed": 0,
                "structures_validated_correct": 0,
                "structures_repaired_or_corrected": 0,
                "communication_events": 0,
                "externalization_events": 0,
                "artifact_consultations": 0,
                "repair_episodes": 0,
                "mismatch_detected": 0,
                "dik_counts": Counter(),
                "construction_externalization_create_events": 0,
                "construction_externalization_revision_events": 0,
                "validation_events": 0,
                "phase_transitions": 0,
                "planning_actions": 0,
                "execution_actions": 0,
                "correction_actions": 0,
                "logistics_wait_events": 0,
                "_seen_projects": set(),
            }
        )

    def _close_phase(self):
        if not self.phase_stats:
            return
        current = self.phase_stats[-1]
        current["end_time"] = round(self.simulation.time, 3)

    def on_event(self, event):
        event_type = event.get("event_type", "unknown")
        payload = event.get("payload_data", {})
        self.events_by_type[event_type] += 1
        self.phase_stats[-1]["events"][event_type] += 1

        reason = payload.get("reason") or payload.get("trigger_reason") or payload.get("failure_category") or payload.get("blocker_category")
        if reason:
            self.reason_distributions[event_type][str(reason)] += 1

        if event_type in {"planner_invocation_requested", "planner_invocation_completed"}:
            trig = payload.get("trigger_reason", "unknown")
            self.breakdown_counts["planner_invocations_by_trigger"][trig] += 1
        if event_type in {"brain_provider_fallback", "brain_provider_timeout", "brain_provider_error"}:
            self.breakdown_counts["planner_fallback_by_reason"][payload.get("reason", "unknown")] += 1
        if event_type in {"brain_provider_response_invalid", "brain_response_rejected"}:
            self.breakdown_counts["invalid_brain_responses"][payload.get("schema_parsing_succeeded", "unknown")] += 1
        if event_type in {"planner_request_skipped_inflight", "planner_request_skipped_cooldown", "planner_request_started_async", "planner_request_completed_async", "planner_request_result_arrived_stale", "planner_response_discarded_due_to_state_change", "backend_degraded_mode_started", "backend_degraded_mode_ended", "ui_safe_fallback_used"}:
            self.breakdown_counts["planner_async_lifecycle"][event_type] += 1
        if event_type == "planner_request_queue_depth":
            depth = int(payload.get("queue_depth", 0) or 0)
            self.breakdown_counts["planner_request_queue_depth"][str(depth)] += 1
        if event_type in {"plan_adopted", "plan_adopted_low_trust", "plan_invalidated"}:
            self.breakdown_counts["plan_method_outcomes"][event_type] += 1
        if event_type.startswith("goal_"):
            self.breakdown_counts["goal_transitions"][event_type] += 1
            gtype = payload.get("goal_type", "unknown")
            self.breakdown_counts["goal_type_adoptions"][gtype] += 1
        if event_type == "action_translation_failed":
            self.breakdown_counts["translation_failures_by_category"][payload.get("failure_category", "unknown")] += 1
        if event_type == "target_resolution_failed":
            self.breakdown_counts["target_resolution_failures_by_category"][payload.get("failure_category", "unknown")] += 1
        if event_type == "execution_readiness_failed":
            self.breakdown_counts["readiness_failures_by_category"][payload.get("failure_category", "unknown")] += 1
        if event_type in {"movement_blocked", "movement_failed"}:
            self.breakdown_counts["movement_failures_by_category"][payload.get("blocker_category", payload.get("failure_category", "unknown"))] += 1
        if event_type in {"stall_started", "stall_continued", "stall_recovered", "repeated_stall_detected"}:
            self.breakdown_counts["stall_events_by_category"][payload.get("stall_reason", "unknown")] += 1
        if event_type in {"repeated_action_loop_detected", "repeated_plan_loop_detected", "repeated_target_failure_detected", "repeated_backend_fallback_detected"}:
            self.breakdown_counts["loop_detections"][event_type] += 1

        if event_type == "communication_exchange":
            for mtype in payload.get("message_types", []):
                self.communication_by_type[mtype] += 1
            self.phase_stats[-1]["communication_events"] += 1

        if event_type in {"externalization_created", "construction_externalization_update"}:
            self.phase_stats[-1]["externalization_events"] += 1

        if event_type == "externalization_created":
            artifact_type = payload.get("type", "unknown")
            self.externalization_by_type[artifact_type] += 1
            agent = payload.get("agent")
            if agent in self.agent_stats:
                self.agent_stats[agent]["plan_externalizations"] += 1


        if event_type == "phase_transition":
            self.phase_stats[-1]["phase_transitions"] += 1
            for stats in self.agent_stats.values():
                stats["phase_transitions_seen"] += 1

        if event_type == "brain_decision_outcome":
            selected = payload.get("selected_action", "")
            planning = {"inspect_information_source", "communicate", "externalize_plan", "consult_team_artifact", "request_assistance"}
            execution = {"transport_resources", "start_construction", "continue_construction"}
            correction = {"repair_or_correct_construction", "validate_construction"}
            agent = payload.get("agent")
            if selected in planning:
                self.phase_stats[-1]["planning_actions"] += 1
                if agent in self.agent_stats:
                    self.agent_stats[agent]["planning_or_externalization_actions"] += 1
            elif selected in execution:
                self.phase_stats[-1]["execution_actions"] += 1
                if agent in self.agent_stats:
                    self.agent_stats[agent]["execution_actions"] += 1
            elif selected in correction:
                self.phase_stats[-1]["correction_actions"] += 1
                if agent in self.agent_stats:
                    self.agent_stats[agent]["correction_validation_actions"] += 1

        if event_type == "construction_waiting_for_logistics":
            self.phase_stats[-1]["logistics_wait_events"] += 1

        if event_type == "construction_externalization_update":
            structure_type = payload.get("structure_type", "unknown")
            project_id = payload.get("project_id")
            self.construction_externalizations_by_type[structure_type] += 1
            if project_id and project_id not in self._seen_construction_projects:
                self._seen_construction_projects.add(project_id)
                self.construction_creation_by_type[structure_type] += 1
                self.phase_stats[-1]["construction_externalization_create_events"] += 1
            else:
                self.construction_revision_by_type[structure_type] += 1
                self.phase_stats[-1]["construction_externalization_revision_events"] += 1
            if project_id:
                self.phase_stats[-1]["_seen_projects"].add(project_id)

            if payload.get("status") == "complete":
                self.phase_stats[-1]["structures_completed"] += 1
            if payload.get("correct") is True:
                self.phase_stats[-1]["structures_validated_correct"] += 1
                self.phase_stats[-1]["validation_events"] += 1
                self.construction_validation_by_type[structure_type] += 1

        if event_type == "artifact_consulted":
            self.phase_stats[-1]["artifact_consultations"] += 1
            agent = payload.get("agent")
            if agent in self.agent_stats:
                self.agent_stats[agent]["artifact_consultations"] += 1

        if event_type == "assistance_requested":
            agent = payload.get("agent")
            if agent in self.agent_stats:
                self.agent_stats[agent]["help_requests"] += 1

        if event_type == "construction_mismatch_detected":
            self.phase_stats[-1]["mismatch_detected"] += 1
            agent = payload.get("agent")
            if agent in self.agent_stats:
                self.agent_stats[agent]["mismatch_detected"] += 1

        if event_type == "construction_repair_episode":
            self.phase_stats[-1]["repair_episodes"] += 1
            self.phase_stats[-1]["structures_repaired_or_corrected"] += 1
            agent = payload.get("agent")
            if agent in self.agent_stats:
                self.agent_stats[agent]["repair_episodes"] += 1

    def on_step(self, dt):
        current_phase_index = self.simulation.environment.current_phase_index
        if current_phase_index != self._last_phase_index:
            self._close_phase()
            self.simulation.logger.log_event(
                self.simulation.time,
                "phase_summary_captured",
                {
                    "phase_name": self.phase_stats[-1]["phase_name"],
                    "start_time": self.phase_stats[-1]["start_time"],
                    "end_time": self.phase_stats[-1]["end_time"],
                },
            )
            self._open_phase()

        for agent in self.simulation.agents:
            entry = self.agent_stats[agent.name]
            self._update_agent_time_buckets(agent, entry, dt)
            self._update_agent_zone(agent, entry, dt)
            status = getattr(agent, "status_last_action", "") or ""
            if "retargeted" in status.lower() and entry["_last_status"] != status:
                entry["retarget_count"] += 1
            entry["_last_status"] = status

        self._track_artifact_validation_transitions()

    def _track_artifact_validation_transitions(self):
        artifacts = self.simulation.team_knowledge_manager.artifacts
        for artifact_id, artifact in artifacts.items():
            prev = self._artifact_state.get(artifact_id)
            curr = artifact.validation_state
            self._artifact_state[artifact_id] = curr
            if prev is not None and prev != curr and curr == "validated":
                self.events_by_type["artifact_validated"] += 1
                self.phase_stats[-1]["validation_events"] += 1

    def _update_agent_zone(self, agent, entry, dt):
        zone_name = self.simulation.environment.get_zone(agent.position)
        zone_key = self.ZONE_MAP.get(zone_name, "transition")
        entry["zone_time"][zone_key] += dt

    def _update_agent_time_buckets(self, agent, entry, dt):
        active = list(getattr(agent, "active_actions", []))
        if not active:
            entry["time_stalled"] += dt
            entry["stall_episode_count"] += 1
            return

        seen_bucket = False
        status = (getattr(agent, "status_last_action", "") or "").lower()
        for action in active:
            atype = action.get("type")
            if atype == "move_to":
                entry["time_moving"] += dt
                seen_bucket = True
                if "source" in status or "inspect" in status:
                    entry["time_inspecting_info"] += dt
            elif atype == "communicate":
                entry["time_communicating"] += dt
                seen_bucket = True
            elif atype == "transport_resources":
                entry["time_transporting"] += dt
                seen_bucket = True
            elif atype == "construct":
                entry["time_constructing"] += dt
                seen_bucket = True
            elif atype == "idle" and action.get("artifact_action") == "externalize_plan":
                entry["time_externalizing"] += dt
                seen_bucket = True
            elif atype == "idle" and action.get("artifact_action") == "consult_team_artifact":
                entry["time_inspecting_info"] += dt
                seen_bucket = True
            elif atype == "idle" and ("repair" in status or "validate" in status):
                entry["time_validating_or_repairing"] += dt
                seen_bucket = True

        if not seen_bucket:
            entry["time_stalled"] += dt
            entry["stall_episode_count"] += 1

    def _collect_dik_counts(self):
        for row in DIK_LOG:
            self.dik_counts[row.get("agent", "unknown")][row.get("type", "Unknown")] += 1
        for name, stats in self.agent_stats.items():
            stats["dik"] = dict(self.dik_counts.get(name, {}))

    def _compute_phase_dik_counts(self):
        for phase in self.phase_stats:
            start = phase.get("start_time", 0.0) or 0.0
            end = phase.get("end_time")
            if end is None:
                end = self.simulation.time
            for row in DIK_LOG:
                t = float(row.get("time", 0.0) or 0.0)
                if start <= t < end:
                    phase["dik_counts"][row.get("type", "Unknown")] += 1

    def _structure_summary(self):
        projects = self.simulation.environment.construction.projects.values()
        summary = {
            "attempted": 0,
            "completed": 0,
            "validated_correct": 0,
            "repaired_or_corrected": self.events_by_type["construction_repair_episode"],
            "by_type": defaultdict(lambda: {"attempted": 0, "completed": 0, "validated_correct": 0}),
        }
        for project in projects:
            ptype = project.get("type", "unknown")
            summary["attempted"] += 1
            summary["by_type"][ptype]["attempted"] += 1
            if project.get("status") == "complete":
                summary["completed"] += 1
                summary["by_type"][ptype]["completed"] += 1
            if project.get("correct", True):
                summary["validated_correct"] += 1
                summary["by_type"][ptype]["validated_correct"] += 1
        summary["by_type"] = dict(summary["by_type"])
        return summary

    def _team_knowledge_summary(self):
        artifacts = self.simulation.team_knowledge_manager.artifacts
        validated = sum(1 for a in artifacts.values() if a.validation_state == "validated")
        revised = self.events_by_type["construction_repair_episode"]
        consulted = self.events_by_type["artifact_consulted"]
        adoption = sum(a.uptake_count for a in artifacts.values())
        overlap = self._team_overlap_index()
        return {
            "artifact_count": len(artifacts),
            "validated_artifacts": validated,
            "artifact_validation_rate": round(validated / len(artifacts), 4) if artifacts else 0.0,
            "artifact_revision_or_repair_count": revised,
            "artifact_consultation_count": consulted,
            "artifact_adoption_count": adoption,
            "team_knowledge_overlap_index": overlap,
            "shared_validated_knowledge_count": len(self.simulation.team_knowledge_manager.validated_knowledge),
        }

    def _team_overlap_index(self):
        rule_sets = []
        for agent in self.simulation.agents:
            rule_sets.append(set(agent.mental_model["knowledge"].rules))
        if not rule_sets:
            return 0.0
        union = set().union(*rule_sets)
        if not union:
            return 0.0
        intersection = set(rule_sets[0])
        for rs in rule_sets[1:]:
            intersection &= rs
        return round(len(intersection) / len(union), 4)


    @staticmethod
    def _top_reasons(counter, n=5):
        return [{"reason": k, "count": v} for k, v in counter.most_common(n)]

    def _run_metadata(self):
        env = self.simulation.environment
        phase_config = [
            {"name": p.get("name"), "duration_minutes": p.get("duration_minutes")}
            for p in (env.phases or [])
        ]
        return {
            "experiment_name": self.simulation.logger.output_session.experiment_name,
            "timestamp": self.simulation.logger.output_session.timestamp,
            "session_folder": str(self.simulation.logger.output_session.session_folder),
            "run_id": getattr(self.simulation, "run_id", None),
            "speed_multiplier": self.simulation.speed_multiplier,
            "flash_mode": self.simulation.flash_mode,
            "seed": getattr(self.simulation, "seed", None),
            "num_agents": len(self.simulation.agents),
            "active_roles": [a.role for a in self.simulation.agents],
            "phase_timing": phase_config,
            "brain_backend": self.simulation.brain_backend_config.backend,
            "configured_brain_backend": getattr(self.simulation, "configured_brain_backend", self.simulation.brain_backend_config.backend),
            "effective_brain_backend": getattr(self.simulation, "effective_brain_backend", self.simulation.brain_backend_config.backend),
            "fallback_backend": self.simulation.brain_backend_config.fallback_backend,
            "local_model_name": self.simulation.brain_backend_config.local_model,
            "local_base_url": self.simulation.brain_backend_config.local_base_url,
            "fallback_occurred": bool(getattr(self.simulation, "fallback_occurred", False)),
            "fallback_count": int(getattr(self.simulation, "backend_fallback_count", 0)),
            "agent_traits": {
                a.name: {
                    "role": a.role,
                    "traits": {
                        "communication_propensity": getattr(a, "communication_propensity", 0.5),
                        "goal_alignment": getattr(a, "goal_alignment", 0.5),
                        "help_tendency": getattr(a, "help_tendency", 0.5),
                        "build_speed": getattr(a, "build_speed", 0.5),
                        "rule_accuracy": getattr(a, "rule_accuracy", 0.5),
                    },
                    "construct_values": dict(getattr(a, "construct_values", {})),
                    "mechanism_profile": dict(getattr(a, "mechanism_profile", {})),
                    "packet_access": list(getattr(a, "allowed_packet", [])),
                }
                for a in self.simulation.agents
            },
        }

    def _phase_summary_rows(self):
        rows = []
        for phase in self.phase_stats:
            rows.append(
                {
                    "phase_name": phase["phase_name"],
                    "phase_index": phase["phase_index"],
                    "start_time": phase["start_time"],
                    "end_time": phase["end_time"],
                    "events": dict(phase["events"]),
                    "breakdown_events": {k: v for k, v in phase["events"].items() if any(k.startswith(p) for p in ["planner_", "brain_", "goal_", "action_translation", "target_resolution", "movement_", "stall_", "repeated_"])},
                    "structures_attempted": len(phase["_seen_projects"]),
                    "structures_completed": phase["structures_completed"],
                    "structures_validated_correct": phase["structures_validated_correct"],
                    "structures_repaired_or_corrected": phase["structures_repaired_or_corrected"],
                    "communication_events": phase["communication_events"],
                    "externalization_events": phase["externalization_events"],
                    "artifact_consultations": phase["artifact_consultations"],
                    "repair_episodes": phase["repair_episodes"],
                    "mismatch_detected": phase["mismatch_detected"],
                    "dik_counts": dict(phase["dik_counts"]),
                    "construction_externalization_create_events": phase["construction_externalization_create_events"],
                    "construction_externalization_revision_events": phase["construction_externalization_revision_events"],
                    "validation_events": phase["validation_events"],
                }
            )
        return rows

    def finalize(self):
        self._close_phase()
        self._collect_dik_counts()
        self._compute_phase_dik_counts()

        for agent in self.simulation.agents:
            self.agent_stats[agent.name]["knowledge_rules_end"] = len(agent.mental_model["knowledge"].rules)

        structure_summary = self._structure_summary()
        team_knowledge_summary = self._team_knowledge_summary()
        planner_states = [getattr(agent, "planner_state", {}) for agent in self.simulation.agents]
        avg_consecutive_failure_streak = 0.0
        streak_samples = sum(int(s.get("consecutive_failure_samples", 0) or 0) for s in planner_states)
        if streak_samples:
            avg_consecutive_failure_streak = round(
                sum(float(s.get("consecutive_failure_sum", 0) or 0.0) for s in planner_states) / streak_samples,
                3,
            )
        planner_responsiveness = {
            "requests_started": int(sum(int(s.get("total_started", 0) or 0) for s in planner_states)),
            "requests_completed": int(sum(int(s.get("total_completed", 0) or 0) for s in planner_states)),
            "requests_timed_out": int(sum(int(s.get("total_timed_out", 0) or 0) for s in planner_states)),
            "requests_failed": int(sum(int(s.get("total_failed", 0) or 0) for s in planner_states)),
            "requests_skipped_due_to_inflight": int(sum(int(s.get("total_skipped_inflight", 0) or 0) for s in planner_states)),
            "requests_skipped_due_to_cooldown": int(sum(int(s.get("total_skipped_cooldown", 0) or 0) for s in planner_states)),
            "degraded_mode_episodes": int(sum(int(s.get("degraded_mode_episodes", 0) or 0) for s in planner_states)),
            "average_consecutive_failure_streak": avg_consecutive_failure_streak,
            "stale_plan_reuse_count": int(sum(int(s.get("stale_plan_reuse_count", 0) or 0) for s in planner_states)),
            "ui_safe_fallback_count": int(sum(int(s.get("ui_safe_fallback_count", 0) or 0) for s in planner_states)),
            "responses_discarded_as_stale": int(sum(int(s.get("total_stale_discarded", 0) or 0) for s in planner_states)),
        }

        run_summary = {
            "run_metadata": self._run_metadata(),
            "outcomes": {
                "total_structures_attempted": structure_summary["attempted"],
                "total_structures_completed": structure_summary["completed"],
                "total_structures_validated_correct": structure_summary["validated_correct"],
                "total_structures_repaired_or_corrected": structure_summary["repaired_or_corrected"],
                "structures_by_type": structure_summary["by_type"],
                "colony_survivability_proxy": {
                    "completed_structures": structure_summary["completed"],
                    "validated_structure_ratio": round(
                        structure_summary["validated_correct"] / structure_summary["attempted"], 4
                    ) if structure_summary["attempted"] else 0.0,
                },
                "phase_objective_completion": {
                    phase["phase_name"]: phase["events"].get("construction_externalization_update", 0) > 0
                    for phase in self.phase_stats
                },
            },
            "process": {
                "dik_acquisition_counts_by_agent": {a: dict(v) for a, v in self.dik_counts.items()},
                "dik_transformation_count": int(sum(v.get("Knowledge", 0) for v in self.dik_counts.values())),
                "communication_counts_by_type": dict(self.communication_by_type),
                "help_requests": self.events_by_type["assistance_requested"],
                "plan_externalizations": self.events_by_type["externalization_created"],
                "artifact_consultations": self.events_by_type["artifact_consulted"],
                "construction_externalization_creations_or_updates": self.events_by_type[
                    "construction_externalization_update"
                ],
                "construction_externalization_creations_by_type": dict(self.construction_creation_by_type),
                "construction_externalization_revisions_by_type": dict(self.construction_revision_by_type),
                "validation_events": self.events_by_type["artifact_validated"],
                "mismatch_detections": self.events_by_type["construction_mismatch_detected"],
                "repair_or_correction_episodes": self.events_by_type["construction_repair_episode"],
                "team_knowledge": team_knowledge_summary,
                "planner_responsiveness": planner_responsiveness,
            },
            "externalization_metrics": {
                "externalized_artifacts_created_by_type": dict(self.externalization_by_type),
                "whiteboard_artifacts_created_by_type": {
                    k: v for k, v in self.externalization_by_type.items() if "whiteboard" in k
                },
                "construction_artifacts_created_by_type": dict(self.construction_externalizations_by_type),
                "construction_artifact_creation_by_type": dict(self.construction_creation_by_type),
                "construction_artifact_revision_by_type": dict(self.construction_revision_by_type),
                "construction_artifact_validation_by_type": dict(self.construction_validation_by_type),
                "artifact_validation_rate": team_knowledge_summary["artifact_validation_rate"],
                "artifact_revision_or_repair_rate": round(
                    self.events_by_type["construction_repair_episode"] / max(1, len(self.simulation.team_knowledge_manager.artifacts)),
                    4,
                ),
                "artifact_consultation_or_use_rate": round(
                    self.events_by_type["artifact_consulted"] / max(1, len(self.simulation.agents)),
                    4,
                ),
                "knowledge_artifact_mismatch_count": self.events_by_type["construction_mismatch_detected"],
                "artifact_uptake_or_adoption_count": team_knowledge_summary["artifact_adoption_count"],
            },
            "behavioral": {
                "time_buckets_by_agent": {
                    name: {
                        key: round(stats[key], 3)
                        for key in [
                            "time_moving",
                            "time_stalled",
                            "time_inspecting_info",
                            "time_communicating",
                            "time_transporting",
                            "time_externalizing",
                            "time_constructing",
                            "time_validating_or_repairing",
                        ]
                    }
                    for name, stats in self.agent_stats.items()
                },
                "zone_occupancy_by_agent": {
                    name: {k: round(v, 3) for k, v in stats["zone_time"].items()}
                    for name, stats in self.agent_stats.items()
                },
                "retarget_counts_by_agent": {name: stats["retarget_count"] for name, stats in self.agent_stats.items()},
                "stall_episode_counts_by_agent": {name: stats["stall_episode_count"] for name, stats in self.agent_stats.items()},
            },
            "events": dict(self.events_by_type),
            "breakdown_metrics": {k: dict(v) for k, v in self.breakdown_counts.items()},
            "reason_top_n": {k: self._top_reasons(v) for k, v in self.reason_distributions.items() if v},
            "end_state": {
                "sim_time": round(self.simulation.time, 3),
                "team_artifact_count": len(self.simulation.team_knowledge_manager.artifacts),
            },
        }

        phase_summary = self._phase_summary_rows()
        agent_rows = []
        for name, stats in self.agent_stats.items():
            row = {
                "agent": name,
                "role": stats["role"],
                "time_moving": round(stats["time_moving"], 3),
                "time_stalled": round(stats["time_stalled"], 3),
                "time_inspecting_info": round(stats["time_inspecting_info"], 3),
                "time_communicating": round(stats["time_communicating"], 3),
                "time_transporting": round(stats["time_transporting"], 3),
                "time_externalizing": round(stats["time_externalizing"], 3),
                "time_constructing": round(stats["time_constructing"], 3),
                "time_validating_or_repairing": round(stats["time_validating_or_repairing"], 3),
                "retarget_count": stats["retarget_count"],
                "stall_episode_count": stats["stall_episode_count"],
                "mismatch_detected": stats["mismatch_detected"],
                "repair_episodes": stats["repair_episodes"],
                "help_requests": stats["help_requests"],
                "plan_externalizations": stats["plan_externalizations"],
                "artifact_consultations": stats["artifact_consultations"],
                "dik_data": stats["dik"].get("Data", 0),
                "dik_information": stats["dik"].get("Information", 0),
                "dik_knowledge": stats["dik"].get("Knowledge", 0),
                "knowledge_rules_end": stats["knowledge_rules_end"],
            }
            agent_rows.append(row)

        team_summary = {
            "team_knowledge": team_knowledge_summary,
            "externalization_metrics": run_summary["externalization_metrics"],
            "communication_counts_by_type": run_summary["process"]["communication_counts_by_type"],
            "events": dict(self.events_by_type),
            "breakdown_metrics": {k: dict(v) for k, v in self.breakdown_counts.items()},
            "reason_top_n": {k: self._top_reasons(v) for k, v in self.reason_distributions.items() if v},
            "backend": {
                "configured_brain_backend": run_summary["run_metadata"].get("configured_brain_backend"),
                "effective_brain_backend": run_summary["run_metadata"].get("effective_brain_backend"),
                "fallback_backend": run_summary["run_metadata"].get("fallback_backend"),
                "fallback_occurred": run_summary["run_metadata"].get("fallback_occurred"),
                "fallback_count": run_summary["run_metadata"].get("fallback_count"),
            },
        }

        self._write_outputs(run_summary, phase_summary, agent_rows, team_summary)
        return run_summary

    def _write_outputs(self, run_summary, phase_summary, agent_rows, team_summary):
        measures_dir = self.simulation.logger.output_session.measures_dir
        measures_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(measures_dir / "run_summary.json", run_summary)
        self._write_json(measures_dir / "phase_summary.json", phase_summary)
        self._write_json(measures_dir / "team_summary.json", team_summary)
        self._write_agent_csv(measures_dir / "agent_summary.csv", agent_rows)

        self.simulation.logger.log_event(
            self.simulation.time,
            "run_summary_saved",
            {"path": str(measures_dir / "run_summary.json")},
        )
        self.simulation.logger.log_event(
            self.simulation.time,
            "phase_summary_saved",
            {"path": str(measures_dir / "phase_summary.json"), "phase_count": len(phase_summary)},
        )
        self.simulation.logger.log_event(
            self.simulation.time,
            "team_summary_saved",
            {"path": str(measures_dir / "team_summary.json")},
        )

    @staticmethod
    def _write_json(path: Path, payload):
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _write_agent_csv(path: Path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
