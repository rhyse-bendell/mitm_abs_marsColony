from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modules.task_validation import TaskValidator


WITNESS_STEP_TAXONOMY = {
    "source_access",
    "data_acquisition",
    "information_derivation",
    "knowledge_derivation",
    "rule_adoption",
    "communication_or_integration",
    "artifact_consultation",
    "readiness_unlock",
    "plan_method_grounded",
    "executable_action_attempted",
    "executable_action_completed",
}

FAILURE_CATEGORIES = {
    "source_not_accessed",
    "inspect_not_started",
    "inspect_not_completed",
    "inspect_completed_dik_not_acquired",
    "inspect_completed_team_dik_acquired",
    "inspect_completed_shared_adoption_not_local",
    "shared_source_access_blocked_by_legality",
    "shared_source_access_blocked_by_mapping",
    "dik_acquired_readiness_not_unlocked",
    "inspect_success_no_new_dik",
    "inspect_success_dik_no_derivation",
    "inspect_success_rule_not_adopted",
    "inspect_success_readiness_blocked_missing_rule",
    "inspect_success_readiness_blocked_missing_target",
    "inspect_success_readiness_blocked_missing_artifact",
    "inspect_success_readiness_blocked_phase",
    "data_not_acquired",
    "derivation_not_triggered",
    "derivation_failed",
    "rule_not_adopted",
    "communication_not_performed",
    "team_integration_missing",
    "readiness_not_unlocked",
    "plan_invalidated",
    "action_translation_failed",
    "target_resolution_failed",
    "movement_not_started",
    "movement_blocked",
    "no_path_found",
    "target_unreachable",
    "blocked_zone",
    "agent_collision_block",
    "arrival_without_progress",
    "execution_not_attempted",
    "unknown",
}


@dataclass
class WitnessStepRuntime:
    raw_step: str
    step_type: str
    status: str = "pending"
    completed_time: Optional[float] = None
    blocked_time: Optional[float] = None
    completed_by: Optional[str] = None
    blocked_by: Optional[str] = None
    details: Optional[Dict[str, object]] = None


class RuntimeWitnessAudit:
    def __init__(self, simulation):
        self.simulation = simulation
        report = TaskValidator().validate(simulation.task_model)
        self.constructive_witnesses = dict(report.constructive_witnesses)
        self.targets: Dict[str, Dict[str, object]] = {}
        self._raw_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._step_type_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._event_guard = False
        self._build_critical_targets()

    @staticmethod
    def _event_agent(payload: Dict[str, object]) -> Optional[str]:
        agent = payload.get("agent")
        return str(agent) if agent is not None else None

    def _target_relevant_to_agent(self, target_id: str, payload: Dict[str, object]) -> bool:
        agent = self._event_agent(payload)
        if not agent:
            return True
        target = self.targets.get(target_id) or {}
        involved = set(target.get("agents_involved") or set())
        if not involved:
            return True
        return agent in involved

    def _block_targets(self, payload: Dict[str, object], failure_category: str, step_hint: Optional[str] = None):
        for tid in self.targets:
            if self._target_relevant_to_agent(tid, payload):
                self._block_target(tid, failure_category, payload, step_hint=step_hint)

    def _canonical_source_id(self, source_id: object) -> Optional[str]:
        if source_id is None:
            return None
        source_text = str(source_id)
        environment = getattr(self.simulation, "environment", None)
        resolver = getattr(environment, "resolve_source_id", None)
        if callable(resolver):
            resolved = resolver(source_text)
            if resolved:
                return str(resolved)
        return source_text

    def _target_ids_for_source(self, source_id: object, payload: Dict[str, object]) -> List[str]:
        canonical = self._canonical_source_id(source_id)
        if not canonical:
            return []
        return [
            tid
            for tid, _ in self._raw_index.get(f"source_access:{canonical}", [])
            if self._target_relevant_to_agent(tid, payload)
        ]

    def _target_ids_for_goal(self, goal_id: object, payload: Dict[str, object]) -> List[str]:
        if goal_id is None:
            return []
        goal_text = str(goal_id)
        return [
            tid
            for tid, _ in self._raw_index.get(f"ground_goal:{goal_text}", [])
            if self._target_relevant_to_agent(tid, payload)
        ]

    def _block_target_ids(self, target_ids: List[str], payload: Dict[str, object], failure_category: str, step_hint: Optional[str] = None):
        for tid in target_ids:
            self._block_target(tid, failure_category, payload, step_hint=step_hint)

    def _build_critical_targets(self):
        critical_goals = {
            g.goal_id
            for g in self.simulation.task_model.goals.values()
            if g.enabled and str(g.goal_level).strip().lower() in {"mission", "phase"}
        }
        critical_rules = {
            rid
            for gid in critical_goals
            for rid in (self.simulation.task_model.goals.get(gid).prerequisite_rules if self.simulation.task_model.goals.get(gid) else [])
        }
        critical_methods = {
            m.method_id
            for m in self.simulation.task_model.plan_methods.values()
            if m.enabled and m.goal_id in critical_goals
        }

        critical_targets = [
            *(f"goal:{gid}" for gid in sorted(critical_goals)),
            *(f"rule:{rid}" for rid in sorted(critical_rules)),
            *(f"method:{mid}" for mid in sorted(critical_methods)),
        ]

        for target_id in critical_targets:
            witness = self.constructive_witnesses.get(target_id, {})
            if not witness.get("constructively_witnessed"):
                continue
            ordered_steps = []
            for idx, raw in enumerate(witness.get("ordered_path", [])):
                stype = self._normalize_step(raw)
                step = WitnessStepRuntime(raw_step=raw, step_type=stype)
                ordered_steps.append(step)
                self._raw_index[raw].append((target_id, idx))
                self._step_type_index[stype].append((target_id, idx))

            self.targets[target_id] = {
                "target_id": target_id,
                "target_type": target_id.split(":", 1)[0],
                "witness_exists_in_validator": True,
                "witness_type": witness.get("witness_type"),
                "validator_summary": {
                    "closure_reachable": witness.get("closure_reachable"),
                    "constructively_witnessed": witness.get("constructively_witnessed"),
                    "ordered_path": list(witness.get("ordered_path", [])),
                    "blockers": list(witness.get("blockers", [])),
                    "phase_scope": witness.get("phase_scope"),
                    "communication_required": witness.get("communication_required", False),
                },
                "ordered_witness_steps": ordered_steps,
                "status": "never_entered",
                "first_failure_step": None,
                "failure_category": None,
                "agents_involved": set(),
                "phase_context": set(),
                "started_time": None,
                "completed_time": None,
            }

    @staticmethod
    def _normalize_step(raw_step: str) -> str:
        raw = str(raw_step or "")
        if raw.startswith("source_access:"):
            return "source_access"
        if raw.startswith("acquire_data:") or raw.startswith("acquire_information:"):
            return "data_acquisition"
        if raw.startswith("derive:"):
            return "information_derivation"
        if raw.startswith("derive_rule:"):
            return "knowledge_derivation"
        if raw.startswith("communicate_integrate:"):
            return "communication_or_integration"
        if raw.startswith("ground_goal:"):
            return "readiness_unlock"
        if raw.startswith("ground_plan_method:"):
            return "plan_method_grounded"
        return "unknown"

    def _emit_witness_event(self, event_type: str, payload: Dict[str, object]):
        if self._event_guard:
            return
        self._event_guard = True
        try:
            self.simulation.logger.log_event(self.simulation.time, event_type, payload)
        finally:
            self._event_guard = False

    def _recover_failed_target_for_step(self, target_id: str, step_type: str, payload: Dict[str, object]) -> bool:
        target = self.targets[target_id]
        if target.get("status") != "failed":
            return False
        steps: List[WitnessStepRuntime] = target["ordered_witness_steps"]
        blocked_step = next((s for s in steps if s.status == "blocked" and s.step_type == step_type), None)
        if blocked_step is None:
            return False
        blocked_step.status = "pending"
        blocked_step.blocked_time = None
        blocked_step.blocked_by = None
        target["status"] = "in_progress"
        target["failure_category"] = None
        target["first_failure_step"] = None
        self._emit_witness_event(
            "source_access_recovered" if step_type == "source_access" else "witness_step_recovered",
            {
                "witness_id": target_id,
                "target_id": target_id,
                "agent": payload.get("agent"),
                "step_type": step_type,
            },
        )
        self._emit_witness_event(
            "witness_step_recovered_after_late_success",
            {
                "witness_id": target_id,
                "target_id": target_id,
                "agent": payload.get("agent"),
                "step_type": step_type,
            },
        )
        return True

    def _recover_shared_source_steps(self, source_id: str, payload: Dict[str, object]):
        for tid, idx in self._raw_index.get(f"source_access:{source_id}", []):
            target = self.targets.get(tid)
            if not target:
                continue
            step = target["ordered_witness_steps"][idx]
            if step.status == "blocked" and step.step_type == "source_access":
                if self._recover_failed_target_for_step(tid, "source_access", payload):
                    self._emit_witness_event(
                        "shared_source_step_recovered",
                        {
                            "witness_id": tid,
                            "target_id": tid,
                            "agent": payload.get("agent"),
                            "source_id": source_id,
                            "step_type": "source_access",
                        },
                    )
                    self._emit_witness_event(
                        "shared_source_step_recovered_after_late_success",
                        {
                            "witness_id": tid,
                            "target_id": tid,
                            "agent": payload.get("agent"),
                            "source_id": source_id,
                            "step_type": "source_access",
                        },
                    )

    def _recover_role_source_steps(self, source_id: str, payload: Dict[str, object]):
        for tid, idx in self._raw_index.get(f"source_access:{source_id}", []):
            target = self.targets.get(tid)
            if not target:
                continue
            step = target["ordered_witness_steps"][idx]
            if step.status == "blocked" and step.step_type == "source_access":
                if self._recover_failed_target_for_step(tid, "source_access", payload):
                    self._emit_witness_event(
                        "role_source_step_recovered_after_late_success",
                        {
                            "witness_id": tid,
                            "target_id": tid,
                            "agent": payload.get("agent"),
                            "source_id": source_id,
                            "step_type": "source_access",
                        },
                    )

    def _complete_step(self, target_id: str, step_index: int, payload: Dict[str, object]):
        target = self.targets[target_id]
        step: WitnessStepRuntime = target["ordered_witness_steps"][step_index]
        if step.status == "completed":
            return

        if target["status"] == "never_entered":
            target["status"] = "in_progress"
            target["started_time"] = self.simulation.time
            self._emit_witness_event("witness_path_started", {"witness_id": target_id, "target_id": target_id, "step_type": step.step_type})

        step.status = "completed"
        step.completed_time = float(self.simulation.time)
        step.completed_by = payload.get("agent")
        step.details = dict(payload)
        if payload.get("agent"):
            target["agents_involved"].add(payload.get("agent"))
        phase = (self.simulation.environment.get_current_phase() or {}).get("name")
        if phase:
            target["phase_context"].add(phase)
        self._emit_witness_event(
            "witness_step_completed",
            {
                "witness_id": target_id,
                "target_id": target_id,
                "agent": payload.get("agent"),
                "step_type": step.step_type,
                "raw_step": step.raw_step,
            },
        )

    def _mark_step_in_progress(self, target_id: str, step_index: int, payload: Dict[str, object]):
        target = self.targets[target_id]
        step: WitnessStepRuntime = target["ordered_witness_steps"][step_index]
        if step.status in {"completed", "blocked", "in_progress"}:
            return
        step.status = "in_progress"
        target["status"] = "in_progress"
        if target.get("started_time") is None:
            target["started_time"] = float(self.simulation.time)
        target["agents_involved"].add(payload.get("agent"))

        if all(s.status == "completed" for s in target["ordered_witness_steps"]):
            target["status"] = "completed"
            target["completed_time"] = float(self.simulation.time)
            self._emit_witness_event("witness_path_completed", {"witness_id": target_id, "target_id": target_id, "agent": payload.get("agent")})

    def _block_target(self, target_id: str, failure_category: str, payload: Dict[str, object], step_hint: Optional[str] = None):
        target = self.targets[target_id]
        if target.get("failure_category"):
            return
        next_step = None
        for idx, step in enumerate(target["ordered_witness_steps"]):
            if step.status != "completed":
                next_step = (idx, step)
                break
        if next_step is None:
            return
        idx, step = next_step
        if step_hint and step.step_type != step_hint:
            return
        step.status = "blocked"
        step.blocked_time = float(self.simulation.time)
        step.blocked_by = payload.get("agent")
        target["status"] = "failed"
        target["failure_category"] = failure_category if failure_category in FAILURE_CATEGORIES else "unknown"
        target["first_failure_step"] = {
            "index": idx,
            "raw_step": step.raw_step,
            "step_type": step.step_type,
            "time": float(self.simulation.time),
        }
        self._emit_witness_event(
            "witness_step_blocked",
            {
                "witness_id": target_id,
                "target_id": target_id,
                "agent": payload.get("agent"),
                "step_type": step.step_type,
                "failure_category": target["failure_category"],
            },
        )
        self._emit_witness_event(
            "witness_path_failed",
            {
                "witness_id": target_id,
                "target_id": target_id,
                "agent": payload.get("agent"),
                "failure_category": target["failure_category"],
            },
        )

    def on_event(self, event: Dict[str, object]):
        event_type = event.get("event_type")
        payload = dict(event.get("payload_data") or {})
        if not event_type or str(event_type).startswith("witness_"):
            return

        if event_type == "source_access_succeeded":
            source_id = payload.get("source_id")
            if source_id:
                if payload.get("is_shared_source"):
                    self._recover_shared_source_steps(source_id, payload)
                else:
                    self._recover_role_source_steps(source_id, payload)
                for tid, idx in self._raw_index.get(f"source_access:{source_id}", []):
                    self._complete_step(tid, idx, payload)
            for element_id in payload.get("new_data_ids", []) + payload.get("new_information_ids", []):
                for prefix in ("acquire_data", "acquire_information"):
                    for tid, idx in self._raw_index.get(f"{prefix}:{element_id}", []):
                        self._complete_step(tid, idx, payload)
            for element_id in payload.get("team_dik_added_ids", []):
                for prefix in ("acquire_data", "acquire_information"):
                    for tid, idx in self._raw_index.get(f"{prefix}:{element_id}", []):
                        self._complete_step(tid, idx, payload)

        elif event_type == "shared_source_access_success":
            source_id = payload.get("source_id")
            if source_id:
                self._recover_shared_source_steps(source_id, payload)
                for tid, idx in self._raw_index.get(f"source_access:{source_id}", []):
                    self._complete_step(tid, idx, payload)

        elif event_type == "inspect_started":
            source_id = payload.get("source_id")
            if source_id:
                for tid, idx in self._raw_index.get(f"source_access:{source_id}", []):
                    self._mark_step_in_progress(tid, idx, payload)

        elif event_type == "inspect_progressed":
            source_id = payload.get("source_id")
            stage = payload.get("stage")
            if source_id and stage in {"target_reached", "inspection_started"}:
                for tid, idx in self._raw_index.get(f"source_access:{source_id}", []):
                    self._mark_step_in_progress(tid, idx, payload)

        elif event_type == "inspect_completion_failed":
            reason = str(payload.get("failure_category") or "inspect_not_completed")
            category = "inspect_completed_dik_not_acquired" if reason == "partial_packet_uptake" else "inspect_not_completed"
            source_targets = self._target_ids_for_source(payload.get("source_id"), payload)
            if source_targets:
                self._block_target_ids(source_targets, payload, category, step_hint="source_access")
            else:
                self._block_targets(payload, category, step_hint="source_access")

        elif event_type == "shared_source_access_blocked":
            reason = str(payload.get("reason") or "blocked")
            classification = str(payload.get("source_access_classification") or "")
            if classification and classification != "shared_team_source":
                return
            category = "shared_source_access_blocked_by_legality" if reason in {"too_far_or_role_mismatch"} else "shared_source_access_blocked_by_mapping"
            source_targets = self._target_ids_for_source(payload.get("source_id"), payload)
            if source_targets:
                self._block_target_ids(source_targets, payload, category, step_hint="source_access")
            else:
                self._block_targets(payload, category, step_hint="source_access")

        elif event_type == "shared_source_dik_acquired_team":
            for element_id in payload.get("added_ids", []):
                for prefix in ("acquire_data", "acquire_information"):
                    for tid, idx in self._raw_index.get(f"{prefix}:{element_id}", []):
                        self._complete_step(tid, idx, payload)

        elif event_type == "inspect_completion_blocked":
            reason = str(payload.get("failure_category") or "inspect_not_completed")
            category = "inspect_not_completed"
            if reason == "readiness_not_unlocked_after_inspect_success":
                category = "dik_acquired_readiness_not_unlocked"
            elif reason in {"target_resolution_failed", "source_reached_inspect_not_started"}:
                category = "inspect_not_started"
            source_targets = self._target_ids_for_source(payload.get("source_id"), payload)
            if source_targets:
                self._block_target_ids(source_targets, payload, category, step_hint="source_access")
            else:
                self._block_targets(payload, category, step_hint="source_access")

        elif event_type == "inspect_post_handoff_classified":
            outcome = str(payload.get("post_inspect_outcome") or "")
            outcome_map = {
                "inspect_success_no_new_dik": "inspect_success_no_new_dik",
                "inspect_success_dik_no_derivation": "inspect_success_dik_no_derivation",
                "inspect_success_rule_not_adopted": "inspect_success_rule_not_adopted",
                "inspect_success_readiness_blocked_missing_rule": "inspect_success_readiness_blocked_missing_rule",
                "inspect_success_readiness_blocked_missing_target": "inspect_success_readiness_blocked_missing_target",
                "inspect_success_readiness_blocked_missing_artifact": "inspect_success_readiness_blocked_missing_artifact",
                "inspect_success_readiness_blocked_phase": "inspect_success_readiness_blocked_phase",
            }
            category = outcome_map.get(outcome)
            if category:
                self._block_targets(payload, category, step_hint="readiness_unlock")

        elif event_type == "dik_derivation_executed":
            did = payload.get("derivation_id")
            out = payload.get("output_element_id")
            if did:
                for tid, idx in self._raw_index.get(f"derive:{did}", []):
                    self._complete_step(tid, idx, payload)
            if out:
                for tid, idx in self._raw_index.get(f"derive_rule:{out}", []):
                    self._complete_step(tid, idx, payload)

        elif event_type == "rule_adopted":
            rule_id = payload.get("rule_id")
            if rule_id:
                for tid, idx in self._raw_index.get(f"derive_rule:{rule_id}", []):
                    self._complete_step(tid, idx, payload)

        elif event_type == "communication_exchange":
            for tid, idx in self._step_type_index.get("communication_or_integration", []):
                self._complete_step(tid, idx, payload)

        elif event_type == "artifact_consulted":
            for tid, idx in self._step_type_index.get("artifact_consultation", []):
                self._complete_step(tid, idx, payload)

        elif event_type == "plan_method_grounding_result":
            method_id = payload.get("plan_method_id")
            status = str(payload.get("plan_method_status") or "")
            if method_id and status not in {"rejected_unknown_method", "rejected_not_grounded", ""}:
                for tid, idx in self._raw_index.get(f"ground_plan_method:{method_id}", []):
                    self._complete_step(tid, idx, payload)

        elif event_type == "execution_readiness_passed":
            goal_id = payload.get("goal_id")
            for tid in self._target_ids_for_goal(goal_id, payload):
                for candidate_tid, idx in self._raw_index.get(f"ground_goal:{goal_id}", []):
                    if candidate_tid == tid:
                        self._complete_step(tid, idx, payload)

        elif event_type == "executable_action_attempted":
            for tid, idx in self._step_type_index.get("executable_action_attempted", []):
                self._complete_step(tid, idx, payload)

        elif event_type == "executable_action_completed":
            for tid, idx in self._step_type_index.get("executable_action_completed", []):
                self._complete_step(tid, idx, payload)

        elif event_type == "source_inspection_blocked":
            source_targets = self._target_ids_for_source(payload.get("source_id"), payload)
            if source_targets:
                self._block_target_ids(source_targets, payload, "inspect_not_completed", step_hint="source_access")
            else:
                self._block_targets(payload, "inspect_not_completed", step_hint="source_access")
        elif event_type == "action_translation_failed":
            source_targets = self._target_ids_for_source(payload.get("source_id") or payload.get("target_id") or payload.get("requested_target_id"), payload)
            if source_targets:
                self._block_target_ids(source_targets, payload, "action_translation_failed")
            else:
                self._block_targets(payload, "action_translation_failed")
        elif event_type == "target_resolution_failed":
            source_targets = self._target_ids_for_source(payload.get("requested_target_id"), payload)
            if source_targets:
                self._block_target_ids(source_targets, payload, "target_resolution_failed")
            else:
                self._block_targets(payload, "target_resolution_failed")
        elif event_type == "movement_blocked":
            blocker = str(payload.get("blocker_category") or "movement_blocked")
            mapping = {
                "no_path_found": "no_path_found",
                "target_unreachable": "target_unreachable",
                "blocked_zone": "blocked_zone",
                "agent_collision_block": "agent_collision_block",
                "arrival_without_progress": "arrival_without_progress",
            }
            failure = mapping.get(blocker, "movement_blocked")
            source_targets = self._target_ids_for_source(payload.get("source_id") or payload.get("target_id"), payload)
            if source_targets:
                self._block_target_ids(source_targets, payload, failure)
            else:
                self._block_targets(payload, failure)
        elif event_type == "plan_invalidated":
            self._block_targets(payload, "plan_invalidated", step_hint="plan_method_grounded")
        elif event_type == "execution_readiness_failed":
            self._block_targets(payload, "readiness_not_unlocked", step_hint="readiness_unlock")

    @staticmethod
    def _default_failure_for_step(step_type: str) -> str:
        mapping = {
            "source_access": "source_not_accessed",
            "data_acquisition": "data_not_acquired",
            "information_derivation": "derivation_not_triggered",
            "knowledge_derivation": "rule_not_adopted",
            "rule_adoption": "rule_not_adopted",
            "communication_or_integration": "communication_not_performed",
            "artifact_consultation": "inspect_not_completed",
            "readiness_unlock": "readiness_not_unlocked",
            "plan_method_grounded": "plan_invalidated",
            "executable_action_attempted": "execution_not_attempted",
            "executable_action_completed": "execution_not_attempted",
        }
        return mapping.get(step_type, "unknown")

    def finalize(self) -> Dict[str, object]:
        failure_counts = Counter()
        completed_by_type = Counter()
        coverage_fracs: List[float] = []
        partial = 0
        zero = 0
        started = 0
        completed = 0
        failed = 0

        audit_rows = []
        for target_id, target in sorted(self.targets.items()):
            steps: List[WitnessStepRuntime] = target["ordered_witness_steps"]
            done = sum(1 for s in steps if s.status == "completed")
            frac = (done / len(steps)) if steps else 0.0
            coverage_fracs.append(frac)
            if done > 0:
                started += 1
            else:
                zero += 1
            if 0 < done < len(steps):
                partial += 1
            if target["status"] == "completed":
                completed += 1
                completed_by_type[target["target_type"]] += 1
            elif target["status"] == "failed":
                failed += 1
            elif done > 0:
                target["status"] = "partial"
            else:
                target["status"] = "never_entered"

            if not target.get("failure_category") and target["status"] != "completed":
                next_step = next((s for s in steps if s.status != "completed"), None)
                if next_step is not None:
                    target["failure_category"] = self._default_failure_for_step(next_step.step_type)
                    target["first_failure_step"] = {
                        "raw_step": next_step.raw_step,
                        "step_type": next_step.step_type,
                    }
            if target.get("failure_category"):
                failure_counts[target["failure_category"]] += 1

            audit_rows.append(
                {
                    "target_id": target_id,
                    "target_type": target["target_type"],
                    "witness_exists_in_validator": target["witness_exists_in_validator"],
                    "witness_type": target.get("witness_type"),
                    "validator_summary": target.get("validator_summary"),
                    "status": target["status"],
                    "coverage_fraction": round(frac, 4),
                    "runtime_completed_steps": [s.raw_step for s in steps if s.status == "completed"],
                    "runtime_uncovered_steps": [s.raw_step for s in steps if s.status != "completed"],
                    "first_failure_step": target.get("first_failure_step"),
                    "failure_category": target.get("failure_category"),
                    "agents_involved": sorted(target["agents_involved"]),
                    "phase_context": sorted(target["phase_context"]),
                    "ordered_witness_steps": [
                        {
                            "raw_step": s.raw_step,
                            "step_type": s.step_type,
                            "status": s.status,
                            "completed_time": s.completed_time,
                            "blocked_time": s.blocked_time,
                            "completed_by": s.completed_by,
                            "blocked_by": s.blocked_by,
                        }
                        for s in steps
                    ],
                }
            )

        summary = {
            "critical_witness_targets_total": len(self.targets),
            "critical_witness_targets_started": started,
            "critical_witness_targets_completed": completed,
            "critical_witness_targets_failed": failed,
            "witness_step_failures_by_category": dict(failure_counts),
            "runtime_witness_partial_count": partial,
            "runtime_witness_zero_progress_count": zero,
            "rule_witnesses_completed": completed_by_type.get("rule", 0),
            "goal_witnesses_completed": completed_by_type.get("goal", 0),
            "method_witnesses_completed": completed_by_type.get("method", 0),
            "average_witness_coverage_fraction": round(sum(coverage_fracs) / max(1, len(coverage_fracs)), 4),
            "top_witness_failure_categories": [
                {"category": k, "count": v} for k, v in failure_counts.most_common(5)
            ],
            "witness_step_taxonomy": sorted(WITNESS_STEP_TAXONOMY),
        }

        out = {
            "task_id": self.simulation.task_model.task_id,
            "step_taxonomy": sorted(WITNESS_STEP_TAXONOMY),
            "failure_categories": sorted(FAILURE_CATEGORIES),
            "critical_targets": audit_rows,
            "summary": summary,
        }
        path = Path(self.simulation.logger.output_session.measures_dir) / "runtime_witness_coverage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        self._emit_witness_event("runtime_witness_audit_saved", {"path": str(path), "target_count": len(audit_rows)})
        return {"artifact_path": str(path), "summary": summary}
