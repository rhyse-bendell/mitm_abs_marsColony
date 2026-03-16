from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
from typing import Dict, List, Optional, Set, Tuple

from modules.task_model import TaskModel, load_task_model


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    context: Dict[str, object] = field(default_factory=dict)


@dataclass
class ReachabilityState:
    reachable: Set[str] = field(default_factory=set)
    provenance: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass
class TaskValidationReport:
    task_id: str
    passed: bool
    issues: List[ValidationIssue]
    reachable_by_role: Dict[str, List[str]]
    reachable_team: List[str]
    unreachable_dik: List[str]
    unreachable_rules: List[str]
    team_only_rules: List[str]
    unsatisfied_goals: List[str]
    unsatisfied_plan_methods: List[str]
    proof_paths: Dict[str, List[str]]
    derivation_edges: List[Tuple[str, str, str]]

    def to_dict(self) -> Dict[str, object]:
        errors = [asdict(i) for i in self.issues if i.severity == "error"]
        warnings = [asdict(i) for i in self.issues if i.severity == "warning"]
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "counts": {
                "errors": len(errors),
                "warnings": len(warnings),
            },
            "errors": errors,
            "warnings": warnings,
            "reachable_by_role": self.reachable_by_role,
            "reachable_team": self.reachable_team,
            "unreachable_dik": self.unreachable_dik,
            "unreachable_rules": self.unreachable_rules,
            "team_only_rules": self.team_only_rules,
            "unsatisfied_goals": self.unsatisfied_goals,
            "unsatisfied_plan_methods": self.unsatisfied_plan_methods,
            "proof_paths": self.proof_paths,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Task Validation Report: `{self.task_id}`",
            "",
            f"**Overall status:** {'PASS' if self.passed else 'FAIL'}",
            f"**Errors:** {len([i for i in self.issues if i.severity == 'error'])}",
            f"**Warnings:** {len([i for i in self.issues if i.severity == 'warning'])}",
            "",
            "## Unreachable DIK",
        ]
        lines.extend([f"- {e}" for e in self.unreachable_dik] or ["- None"])
        lines.extend(["", "## Unreachable Rules"])
        lines.extend([f"- {r}" for r in self.unreachable_rules] or ["- None"])
        lines.extend(["", "## Team-only Rules (need communication/integration)"])
        lines.extend([f"- {r}" for r in self.team_only_rules] or ["- None"])
        lines.extend(["", "## Unsatisfied Goals"])
        lines.extend([f"- {g}" for g in self.unsatisfied_goals] or ["- None"])
        lines.extend(["", "## Unsatisfied Plan Methods"])
        lines.extend([f"- {m}" for m in self.unsatisfied_plan_methods] or ["- None"])
        lines.extend(["", "## Minimal Proof Paths (sample)"])
        if self.proof_paths:
            for rid, path in sorted(self.proof_paths.items()):
                lines.append(f"- `{rid}`: {' -> '.join(path)}")
        else:
            lines.append("- None")

        lines.extend(["", "## Diagnostics"])
        if self.issues:
            for issue in self.issues:
                lines.append(f"- **{issue.severity.upper()}** `{issue.code}`: {issue.message}")
        else:
            lines.append("- No issues found.")
        return "\n".join(lines) + "\n"

    def write_artifacts(self, output_dir: str | Path) -> Dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "task_validation_report.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

        md_path = out_dir / "task_validation_report.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")

        edges_path = out_dir / "dik_derivation_edges.csv"
        with edges_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["from", "to", "edge_type"])
            for src, dst, edge_type in self.derivation_edges:
                writer.writerow([src, dst, edge_type])

        return {
            "json": json_path,
            "markdown": md_path,
            "edges_csv": edges_path,
        }


class TaskValidator:
    def validate(self, task_model: TaskModel) -> TaskValidationReport:
        issues: List[ValidationIssue] = []
        issues.extend(self._check_duplicate_ids(task_model.base_path))
        issues.extend(self._check_referential_integrity(task_model))

        role_states = self._compute_reachability(task_model)
        team_state = self._compute_team_reachability(task_model)

        enabled_dik = {eid for eid, e in task_model.dik_elements.items() if e.enabled}
        unreachable_dik = sorted(enabled_dik - team_state.reachable)

        reachable_rules_team, rule_proofs = self._reachable_rules(task_model, team_state.reachable)
        unreachable_rules = sorted(
            rule_id
            for rule_id, rule in task_model.rules.items()
            if rule.enabled and rule_id not in reachable_rules_team
        )

        team_only_rules = sorted(
            rule_id
            for rule_id in reachable_rules_team
            if not any(rule_id in self._reachable_rules(task_model, rs.reachable)[0] for rs in role_states.values())
        )

        unsatisfied_goals = sorted(
            goal.goal_id
            for goal in task_model.goals.values()
            if goal.enabled and not set(goal.prerequisite_rules).issubset(reachable_rules_team)
        )

        unsatisfied_plan_methods = []
        for method in task_model.plan_methods.values():
            if not method.enabled:
                continue
            missing = (
                set(method.required_rules) - reachable_rules_team
                or set(method.required_knowledge) - team_state.reachable
                or set(method.required_information) - team_state.reachable
                or set(method.required_data) - team_state.reachable
            )
            if missing:
                unsatisfied_plan_methods.append(method.method_id)

        for rule_id in unreachable_rules:
            rule = task_model.rules[rule_id]
            missing_k = sorted(k for k in rule.required_knowledge if k not in team_state.reachable)
            missing_i = sorted(i for i in rule.required_information if i not in team_state.reachable)
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="RULE_UNREACHABLE",
                    message=f"Rule '{rule_id}' is unreachable from configured DIK sources/derivations.",
                    context={"missing_knowledge": missing_k, "missing_information": missing_i},
                )
            )

        for goal_id in unsatisfied_goals:
            goal = task_model.goals[goal_id]
            missing_rules = sorted(r for r in goal.prerequisite_rules if r not in reachable_rules_team)
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="GOAL_UNSATISFIED",
                    message=f"Goal '{goal_id}' has unreachable prerequisite rules.",
                    context={"missing_rules": missing_rules},
                )
            )

        for method_id in unsatisfied_plan_methods:
            method = task_model.plan_methods[method_id]
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="PLAN_METHOD_UNGROUNDED",
                    message=f"Plan method '{method_id}' has unreachable prerequisites.",
                    context={
                        "missing_rules": sorted(r for r in method.required_rules if r not in reachable_rules_team),
                        "missing_knowledge": sorted(k for k in method.required_knowledge if k not in team_state.reachable),
                        "missing_information": sorted(i for i in method.required_information if i not in team_state.reachable),
                        "missing_data": sorted(d for d in method.required_data if d not in team_state.reachable),
                    },
                )
            )

        for method in task_model.plan_methods.values():
            if not method.enabled:
                continue
            action_steps: List[str] = []
            for raw_step in method.candidate_steps:
                action_steps.extend([s.strip() for s in raw_step.split('>') if s.strip()])
            for step in action_steps:
                action = task_model.action_availability.get(step)
                if action is None:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="UNKNOWN_ACTION",
                            message=f"Plan method '{method.method_id}' references unknown action '{step}'.",
                        )
                    )
                elif not action.enabled:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="DISABLED_ACTION",
                            message=f"Plan method '{method.method_id}' uses disabled action '{step}'.",
                        )
                    )

        mission_goal_ids = {
            g.goal_id
            for g in task_model.goals.values()
            if g.enabled and str(g.goal_level).strip().lower() in {"mission", "phase"}
        }
        reachable_goals = {
            g.goal_id
            for g in task_model.goals.values()
            if g.enabled and set(g.prerequisite_rules).issubset(reachable_rules_team)
        }
        unresolved_critical_goals = sorted(mission_goal_ids - reachable_goals)
        if unresolved_critical_goals:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="IDEALIZED_TEAM_UNSOLVABLE",
                    message="Idealized team solvability failed for critical mission/phase goals.",
                    context={"missing_goals": unresolved_critical_goals},
                )
            )

        derivation_edges = self._derivation_edges(task_model)

        errors = [i for i in issues if i.severity == "error"]
        return TaskValidationReport(
            task_id=task_model.task_id,
            passed=not errors,
            issues=issues,
            reachable_by_role={role: sorted(state.reachable) for role, state in role_states.items()},
            reachable_team=sorted(team_state.reachable),
            unreachable_dik=unreachable_dik,
            unreachable_rules=unreachable_rules,
            team_only_rules=team_only_rules,
            unsatisfied_goals=unsatisfied_goals,
            unsatisfied_plan_methods=sorted(unsatisfied_plan_methods),
            proof_paths={k: v for k, v in sorted(rule_proofs.items())},
            derivation_edges=derivation_edges,
        )

    def _check_duplicate_ids(self, base_path: Path) -> List[ValidationIssue]:
        id_columns = {
            "task_sources.csv": "source_id",
            "dik_elements.csv": "element_id",
            "dik_derivations.csv": "derivation_id",
            "rule_definitions.csv": "rule_id",
            "goal_definitions.csv": "goal_id",
            "plan_methods.csv": "method_id",
            "artifact_definitions.csv": "artifact_type",
            "role_definitions.csv": "role_id",
            "action_availability.csv": "action_id",
            "construction_templates.csv": "project_id",
        }
        issues: List[ValidationIssue] = []
        for filename, col in id_columns.items():
            path = base_path / filename
            seen: Set[str] = set()
            dups: Set[str] = set()
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    value = (row.get(col) or "").strip()
                    if not value:
                        continue
                    if value in seen:
                        dups.add(value)
                    seen.add(value)
            for dup in sorted(dups):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="DUPLICATE_ID",
                        message=f"Duplicate id '{dup}' in {filename} ({col}).",
                    )
                )
        return issues

    def _check_referential_integrity(self, task_model: TaskModel) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        valid_elem_types = {"data", "information", "knowledge"}

        for row in task_model.source_contents:
            if row.source_id not in task_model.sources:
                issues.append(ValidationIssue("error", "MISSING_SOURCE", f"source_contents references unknown source '{row.source_id}'."))
            element = task_model.dik_elements.get(row.element_id)
            if element is None:
                issues.append(ValidationIssue("error", "MISSING_DIK_ELEMENT", f"source_contents references unknown element '{row.element_id}'."))
            elif element.element_type != row.element_type:
                issues.append(
                    ValidationIssue(
                        "error",
                        "DIK_TYPE_MISMATCH",
                        f"source_contents element_type mismatch for '{row.element_id}': row={row.element_type}, element={element.element_type}.",
                    )
                )

        for elem in task_model.dik_elements.values():
            normalized_source = (elem.canonical_source or '').strip().lower()
            if elem.enabled and elem.canonical_source and normalized_source not in {'derived', 'inferred'} and elem.canonical_source not in task_model.sources:
                issues.append(ValidationIssue("error", "MISSING_CANONICAL_SOURCE", f"DIK element '{elem.element_id}' references unknown canonical source '{elem.canonical_source}'."))
            if elem.element_type not in valid_elem_types:
                issues.append(ValidationIssue("error", "INVALID_DIK_TYPE", f"DIK element '{elem.element_id}' has invalid type '{elem.element_type}'."))

        for d in task_model.derivations.values():
            if d.output_element_id not in task_model.dik_elements:
                issues.append(ValidationIssue("error", "MISSING_DERIVATION_OUTPUT", f"Derivation '{d.derivation_id}' output '{d.output_element_id}' not in dik_elements."))
            if d.output_type not in valid_elem_types:
                issues.append(ValidationIssue("error", "INVALID_DERIVATION_OUTPUT_TYPE", f"Derivation '{d.derivation_id}' has invalid output_type '{d.output_type}'."))
            all_inputs = set(d.required_inputs) | set(d.optional_inputs)
            for inp in all_inputs:
                if inp not in task_model.dik_elements:
                    issues.append(ValidationIssue("error", "MISSING_DERIVATION_INPUT", f"Derivation '{d.derivation_id}' references unknown input '{inp}'."))
            if d.min_required_count < 0 or d.min_required_count > len(all_inputs):
                issues.append(
                    ValidationIssue(
                        "error",
                        "INVALID_MIN_REQUIRED_COUNT",
                        f"Derivation '{d.derivation_id}' has invalid min_required_count={d.min_required_count} for {len(all_inputs)} total inputs.",
                    )
                )

        for rule in task_model.rules.values():
            for kid in rule.required_knowledge:
                e = task_model.dik_elements.get(kid)
                if e is None:
                    issues.append(ValidationIssue("error", "MISSING_RULE_KNOWLEDGE", f"Rule '{rule.rule_id}' references unknown knowledge '{kid}'."))
                elif e.element_type != "knowledge":
                    issues.append(ValidationIssue("error", "RULE_KNOWLEDGE_TYPE_MISMATCH", f"Rule '{rule.rule_id}' requires '{kid}' but element is type '{e.element_type}'."))
            for iid in rule.required_information:
                e = task_model.dik_elements.get(iid)
                if e is None:
                    issues.append(ValidationIssue("error", "MISSING_RULE_INFORMATION", f"Rule '{rule.rule_id}' references unknown information '{iid}'."))
                elif e.element_type != "information":
                    issues.append(ValidationIssue("error", "RULE_INFORMATION_TYPE_MISMATCH", f"Rule '{rule.rule_id}' requires '{iid}' but element is type '{e.element_type}'."))

        for goal in task_model.goals.values():
            for rid in goal.prerequisite_rules:
                if rid not in task_model.rules:
                    issues.append(ValidationIssue("error", "MISSING_GOAL_RULE", f"Goal '{goal.goal_id}' references unknown prerequisite rule '{rid}'."))

        for method in task_model.plan_methods.values():
            if method.goal_id not in task_model.goals:
                issues.append(ValidationIssue("error", "MISSING_METHOD_GOAL", f"Plan method '{method.method_id}' references unknown goal '{method.goal_id}'."))
            for rid in method.required_rules:
                if rid not in task_model.rules:
                    issues.append(ValidationIssue("error", "MISSING_METHOD_RULE", f"Plan method '{method.method_id}' requires unknown rule '{rid}'."))
            for bucket, ids, expected in (
                ("knowledge", method.required_knowledge, "knowledge"),
                ("information", method.required_information, "information"),
                ("data", method.required_data, "data"),
            ):
                for eid in ids:
                    e = task_model.dik_elements.get(eid)
                    if e is None:
                        issues.append(ValidationIssue("error", "MISSING_METHOD_DIK", f"Plan method '{method.method_id}' requires unknown {bucket} element '{eid}'."))
                    elif e.element_type != expected:
                        issues.append(ValidationIssue("error", "METHOD_DIK_TYPE_MISMATCH", f"Plan method '{method.method_id}' expects {expected} '{eid}' but got '{e.element_type}'."))

        for role in task_model.roles.values():
            if role.spawn_id and role.spawn_id not in {s.spawn_id for s in task_model.spawn_points}:
                issues.append(ValidationIssue("error", "MISSING_ROLE_SPAWN", f"Role '{role.role_id}' references unknown spawn_id '{role.spawn_id}'."))
            for sid in role.source_scope:
                if sid not in task_model.sources:
                    issues.append(ValidationIssue("error", "MISSING_ROLE_SOURCE_SCOPE", f"Role '{role.role_id}' source_scope contains unknown source '{sid}'."))

        for default in task_model.agent_defaults:
            if default.role_id not in task_model.roles:
                issues.append(ValidationIssue("error", "MISSING_AGENT_DEFAULT_ROLE", f"Agent default '{default.agent_name}' references unknown role '{default.role_id}'."))
            for sid in default.source_access_override:
                if sid not in task_model.sources:
                    issues.append(ValidationIssue("error", "MISSING_AGENT_SOURCE_OVERRIDE", f"Agent '{default.agent_name}' references unknown source override '{sid}'."))

        for target in task_model.interaction_targets.values():
            if target.zone_id and target.zone_id not in task_model.zones:
                issues.append(ValidationIssue("error", "MISSING_TARGET_ZONE", f"Interaction target '{target.target_id}' references unknown zone '{target.zone_id}'."))
            if target.object_id and target.object_id not in task_model.environment_objects:
                issues.append(ValidationIssue("error", "MISSING_TARGET_OBJECT", f"Interaction target '{target.target_id}' references unknown object '{target.object_id}'."))

        for node in task_model.resource_nodes.values():
            if node.zone_id and node.zone_id not in task_model.zones:
                issues.append(ValidationIssue("error", "MISSING_RESOURCE_ZONE", f"Resource node '{node.node_id}' references unknown zone '{node.zone_id}'."))

        for template in task_model.construction_templates.values():
            if template.target_id not in task_model.interaction_targets:
                issues.append(ValidationIssue("error", "MISSING_CONSTRUCTION_TARGET", f"Construction template '{template.project_id}' references unknown target_id '{template.target_id}'."))
            if template.artifact_type and template.artifact_type not in task_model.artifacts:
                issues.append(ValidationIssue("error", "MISSING_CONSTRUCTION_ARTIFACT", f"Construction template '{template.project_id}' uses unknown artifact_type '{template.artifact_type}'."))
            for token in template.expected_rules:
                if token.startswith("rule:"):
                    continue
                if token and token not in task_model.rules:
                    issues.append(ValidationIssue("warning", "UNKNOWN_EXPECTED_RULE_TOKEN", f"Construction template '{template.project_id}' has unrecognized expected_rules token '{token}'."))

        for action_id in task_model.action_parameters:
            if action_id not in task_model.action_availability:
                issues.append(ValidationIssue("error", "MISSING_ACTION_AVAILABILITY", f"Action parameters reference unknown action '{action_id}'."))

        return issues

    def _sources_by_role(self, task_model: TaskModel) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for role_id in task_model.roles.keys():
            allowed = set(task_model.source_ids_for_role(role_id))
            role = task_model.roles.get(role_id)
            if role and role.source_scope:
                allowed |= set(role.source_scope)
            out[role_id] = {sid for sid in allowed if sid in task_model.sources and task_model.sources[sid].enabled}

        for default in task_model.agent_defaults:
            if default.source_access_override:
                out.setdefault(default.role_id, set()).update(
                    sid for sid in default.source_access_override if sid in task_model.sources and task_model.sources[sid].enabled
                )
        return out

    def _initial_reachable_from_sources(self, task_model: TaskModel, source_ids: Set[str]) -> ReachabilityState:
        state = ReachabilityState()
        for row in task_model.source_contents:
            if not row.enabled or row.source_id not in source_ids:
                continue
            element = task_model.dik_elements.get(row.element_id)
            if element is None or not element.enabled:
                continue
            state.reachable.add(row.element_id)
            state.provenance.setdefault(
                row.element_id,
                {"kind": "source", "source_id": row.source_id, "inputs": []},
            )
        return state

    def _apply_derivation_closure(self, task_model: TaskModel, state: ReachabilityState) -> ReachabilityState:
        changed = True
        while changed:
            changed = False
            for derivation in task_model.derivations.values():
                if not derivation.enabled:
                    continue
                output = derivation.output_element_id
                if output in state.reachable:
                    continue
                required = set(derivation.required_inputs)
                optional = set(derivation.optional_inputs)
                all_inputs = required | optional
                if required and not required.issubset(state.reachable):
                    continue
                if derivation.min_required_count > 0 and len(all_inputs & state.reachable) < derivation.min_required_count:
                    continue
                state.reachable.add(output)
                state.provenance[output] = {
                    "kind": "derivation",
                    "derivation_id": derivation.derivation_id,
                    "inputs": sorted(all_inputs & state.reachable),
                }
                changed = True
        return state

    def _compute_reachability(self, task_model: TaskModel) -> Dict[str, ReachabilityState]:
        states: Dict[str, ReachabilityState] = {}
        for role_id, sources in self._sources_by_role(task_model).items():
            state = self._initial_reachable_from_sources(task_model, sources)
            states[role_id] = self._apply_derivation_closure(task_model, state)
        return states

    def _compute_team_reachability(self, task_model: TaskModel) -> ReachabilityState:
        team_sources = {sid for sid, src in task_model.sources.items() if src.enabled}
        state = self._initial_reachable_from_sources(task_model, team_sources)
        return self._apply_derivation_closure(task_model, state)

    def _reachable_rules(self, task_model: TaskModel, reachable_elements: Set[str]) -> Tuple[Set[str], Dict[str, List[str]]]:
        rules: Set[str] = set()
        proofs: Dict[str, List[str]] = {}
        for rule in task_model.rules.values():
            if not rule.enabled:
                continue
            needed = set(rule.required_knowledge) | set(rule.required_information)
            if needed.issubset(reachable_elements):
                rules.add(rule.rule_id)
                proof = []
                for eid in sorted(needed):
                    proof.extend(self._trace_element_path(eid, task_model, reachable_elements))
                proofs[rule.rule_id] = list(dict.fromkeys(proof + [f"rule:{rule.rule_id}"]))
        return rules, proofs

    def _trace_element_path(self, element_id: str, task_model: TaskModel, reachable_elements: Set[str]) -> List[str]:
        if element_id not in reachable_elements:
            return [f"missing:{element_id}"]
        for row in task_model.source_contents:
            if row.enabled and row.element_id == element_id and row.source_id in task_model.sources:
                if task_model.sources[row.source_id].enabled:
                    return [f"source:{row.source_id}", f"dik:{element_id}"]
        candidates = [d for d in task_model.derivations.values() if d.enabled and d.output_element_id == element_id]
        if not candidates:
            return [f"dik:{element_id}"]
        best = min(candidates, key=lambda d: len(d.required_inputs) + len(d.optional_inputs))
        out: List[str] = []
        for inp in sorted(set(best.required_inputs + best.optional_inputs)):
            if inp in reachable_elements:
                out.extend(self._trace_element_path(inp, task_model, reachable_elements))
        out.append(f"derivation:{best.derivation_id}")
        out.append(f"dik:{element_id}")
        return out

    def _derivation_edges(self, task_model: TaskModel) -> List[Tuple[str, str, str]]:
        edges: List[Tuple[str, str, str]] = []
        for row in task_model.source_contents:
            if row.enabled:
                edges.append((f"source:{row.source_id}", f"dik:{row.element_id}", "source_content"))
        for derivation in task_model.derivations.values():
            if not derivation.enabled:
                continue
            for src in sorted(set(derivation.required_inputs + derivation.optional_inputs)):
                edges.append((f"dik:{src}", f"dik:{derivation.output_element_id}", f"derivation:{derivation.derivation_id}"))
        return edges


def validate_task_model(task_model: TaskModel) -> TaskValidationReport:
    return TaskValidator().validate(task_model)


def run_task_validation(
    task_id: str = "mars_colony",
    *,
    config_root: str | Path = "config/tasks",
    output_dir: Optional[str | Path] = None,
) -> TaskValidationReport:
    model = load_task_model(task_id=task_id, config_root=config_root)
    report = validate_task_model(model)
    if output_dir is not None:
        report.write_artifacts(output_dir)
    return report
