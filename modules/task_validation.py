from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
from typing import Dict, List, Optional, Set, Tuple

from modules.task_model import TaskModel, load_task_model, normalize_rule_token


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
    constructive_witnesses: Dict[str, Dict[str, object]]

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
            "constructive_witnesses": self.constructive_witnesses,
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

        lines.extend(["", "## Constructive Witness Summary"])
        if self.constructive_witnesses:
            for target_id, witness in sorted(self.constructive_witnesses.items()):
                lines.append(
                    "- `{}`: closure_reachable={}, constructively_witnessed={}, type={}".format(
                        target_id,
                        witness.get("closure_reachable"),
                        witness.get("constructively_witnessed"),
                        witness.get("witness_type", "n/a"),
                    )
                )
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

        constructive_json_path = out_dir / "constructive_witness_report.json"
        constructive_json_path.write_text(
            json.dumps(
                {
                    "task_id": self.task_id,
                    "constructive_witnesses": self.constructive_witnesses,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        constructive_md_path = out_dir / "constructive_witness_report.md"
        lines = [f"# Constructive Witness Report: `{self.task_id}`", ""]
        for target_id, witness in sorted(self.constructive_witnesses.items()):
            lines.append(f"## {target_id}")
            lines.append(f"- closure_reachable: {witness.get('closure_reachable')}")
            lines.append(f"- constructively_witnessed: {witness.get('constructively_witnessed')}")
            lines.append(f"- witness_type: {witness.get('witness_type', 'n/a')}")
            lines.append(f"- communication_required: {witness.get('communication_required', False)}")
            lines.append(f"- phase_constrained: {witness.get('phase_constrained', False)}")
            lines.append(f"- derivation_depth: {witness.get('derivation_depth', 0)}")
            lines.append(f"- upstream_dependency_count: {witness.get('upstream_dependency_count', 0)}")
            blockers = witness.get("blockers") or []
            lines.append("- blockers:")
            lines.extend([f"  - {b}" for b in blockers] or ["  - none"])
            ordered_path = witness.get("ordered_path") or []
            lines.append("- ordered_path:")
            for step in ordered_path:
                lines.append(f"  - {step}")
            if not ordered_path:
                lines.append("  - none")
            lines.append("")
        constructive_md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

        return {
            "json": json_path,
            "markdown": md_path,
            "edges_csv": edges_path,
            "constructive_json": constructive_json_path,
            "constructive_markdown": constructive_md_path,
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
        constructive_witnesses = self._build_constructive_witnesses(
            task_model,
            role_states,
            team_state,
            reachable_rules_team,
        )
        for target_id, witness in constructive_witnesses.items():
            if witness.get("closure_reachable") and not witness.get("constructively_witnessed"):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="CLOSURE_ONLY_REACHABLE",
                        message=f"{target_id} is closure-reachable but lacks an ordered constructive witness.",
                        context={"blockers": witness.get("blockers", [])},
                    )
                )

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
            constructive_witnesses={k: v for k, v in sorted(constructive_witnesses.items())},
        )

    def _phase_compatible(self, scope: str, target_phase: str) -> bool:
        s = (scope or "all").strip().lower()
        p = (target_phase or "all").strip().lower()
        return s in {"", "all", p}

    def _roles_for_element(self, role_states: Dict[str, ReachabilityState], element_id: str) -> Set[str]:
        return {role_id for role_id, state in role_states.items() if element_id in state.reachable}

    def _element_witness(
        self,
        task_model: TaskModel,
        element_id: str,
        reachable_elements: Set[str],
        phase_scope: str,
        visited: Optional[Set[str]] = None,
    ) -> Dict[str, object]:
        seen = set(visited or set())
        if element_id in seen:
            return {"ok": False, "steps": [], "depth": 0, "deps": set(), "blockers": [f"cycle:{element_id}"]}
        seen.add(element_id)

        element = task_model.dik_elements.get(element_id)
        if element is None or not element.enabled:
            return {"ok": False, "steps": [], "depth": 0, "deps": set(), "blockers": [f"missing_element:{element_id}"]}
        if element_id not in reachable_elements:
            return {"ok": False, "steps": [], "depth": 0, "deps": set(), "blockers": [f"unreachable_element:{element_id}"]}
        if not self._phase_compatible(element.phase_scope, phase_scope):
            return {"ok": False, "steps": [], "depth": 0, "deps": set(), "blockers": [f"phase_mismatch:{element_id}"]}

        source_steps = []
        for row in task_model.source_contents:
            if not row.enabled or row.element_id != element_id:
                continue
            src = task_model.sources.get(row.source_id)
            if src is None or not src.enabled:
                continue
            source_steps.append(
                {
                    "ok": True,
                    "steps": [f"source_access:{row.source_id}", f"acquire_{element.element_type}:{element_id}"],
                    "depth": 0,
                    "deps": {element_id},
                    "blockers": [],
                }
            )
        if source_steps:
            return source_steps[0]

        best: Optional[Dict[str, object]] = None
        blockers: List[str] = []
        for d in task_model.derivations.values():
            if not d.enabled or d.output_element_id != element_id:
                continue
            inputs = sorted(set(d.required_inputs + d.optional_inputs) & reachable_elements)
            if set(d.required_inputs) - set(inputs):
                blockers.append(f"missing_required_inputs:{d.derivation_id}")
                continue
            if d.min_required_count > 0 and len(inputs) < d.min_required_count:
                blockers.append(f"min_required_count_unsatisfied:{d.derivation_id}")
                continue
            step_lists: List[str] = []
            max_depth = 0
            deps: Set[str] = {element_id}
            derivation_blockers: List[str] = []
            ok = True
            for inp in inputs:
                sub = self._element_witness(task_model, inp, reachable_elements, phase_scope, seen)
                if not sub["ok"]:
                    ok = False
                    derivation_blockers.extend(sub["blockers"])
                    continue
                step_lists.extend(sub["steps"])
                max_depth = max(max_depth, int(sub["depth"]))
                deps |= set(sub["deps"])
            if not ok:
                blockers.extend(derivation_blockers)
                continue
            candidate = {
                "ok": True,
                "steps": list(dict.fromkeys(step_lists + [f"derive:{d.derivation_id}->{element_id}"])),
                "depth": max_depth + 1,
                "deps": deps,
                "blockers": [],
            }
            if best is None or len(candidate["steps"]) < len(best["steps"]):
                best = candidate

        return best or {"ok": False, "steps": [], "depth": 0, "deps": {element_id}, "blockers": sorted(set(blockers or [f"no_derivation:{element_id}"]))}

    def _build_constructive_witnesses(
        self,
        task_model: TaskModel,
        role_states: Dict[str, ReachabilityState],
        team_state: ReachabilityState,
        reachable_rules_team: Set[str],
    ) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        enabled_rules = {r.rule_id for r in task_model.rules.values() if r.enabled}
        critical_goal_ids = {
            g.goal_id for g in task_model.goals.values() if g.enabled and str(g.goal_level).strip().lower() in {"mission", "phase"}
        }
        critical_rule_ids: Set[str] = set()
        for gid in critical_goal_ids:
            critical_rule_ids |= set(task_model.goals[gid].prerequisite_rules)
        for method in task_model.plan_methods.values():
            if method.enabled and (method.goal_id in critical_goal_ids or str(method.phase_scope).strip().lower() in {"planning", "phase1", "phase2"}):
                critical_rule_ids |= set(method.required_rules)
        for method in task_model.plan_methods.values():
            if method.enabled:
                critical_rule_ids |= set(method.required_rules)
        critical_rule_ids &= enabled_rules

        for rule_id in sorted(critical_rule_ids):
            rule = task_model.rules[rule_id]
            needed = sorted(set(rule.required_knowledge + rule.required_information))
            phase_scope = rule.phase_scope or "all"
            closure_reachable = rule_id in reachable_rules_team
            team_parts = [self._element_witness(task_model, eid, team_state.reachable, phase_scope) for eid in needed]
            constructively_witnessed = closure_reachable and all(p["ok"] for p in team_parts)
            roles_by_req = {eid: self._roles_for_element(role_states, eid) for eid in needed}
            individual_roles = [role for role, state in role_states.items() if set(needed).issubset(state.reachable)]
            communication_required = not individual_roles and constructively_witnessed and bool(needed)
            witness_type = "individual" if individual_roles else ("team-only" if constructively_witnessed else "unwitnessed")
            ordered = []
            for part in team_parts:
                ordered.extend(part.get("steps", []))
            if communication_required:
                ordered.append("communicate_integrate:cross_role_dependency")
            if constructively_witnessed:
                ordered.append(f"derive_rule:{rule_id}")
            deps = set().union(*[set(p.get("deps", set())) for p in team_parts]) if team_parts else set()
            depth = max([int(p.get("depth", 0)) for p in team_parts] or [0])
            blockers = sorted(set(b for p in team_parts for b in p.get("blockers", [])))
            out[f"rule:{rule_id}"] = {
                "closure_reachable": closure_reachable,
                "constructively_witnessed": constructively_witnessed,
                "witness_type": witness_type,
                "phase_constrained": str(phase_scope).strip().lower() not in {"", "all"},
                "phase_scope": phase_scope,
                "communication_required": communication_required,
                "derivation_depth": depth,
                "upstream_dependency_count": len(deps),
                "shared_or_team_only_dependencies": sorted(e for e in deps if task_model.dik_elements.get(e) and task_model.dik_elements[e].role_scope in {"team", "shared"}),
                "brittle_chain": depth >= 4 or len(deps) >= 8,
                "ordered_path": list(dict.fromkeys(ordered)),
                "blockers": blockers,
                "roles_covering_requirements": {k: sorted(v) for k, v in roles_by_req.items()},
            }

        for goal in task_model.goals.values():
            if not goal.enabled or goal.goal_id not in critical_goal_ids:
                continue
            needed_rules = sorted(goal.prerequisite_rules)
            closure_reachable = set(needed_rules).issubset(reachable_rules_team)
            witness_refs = [out.get(f"rule:{rid}") for rid in needed_rules]
            constructively_witnessed = closure_reachable and all(w and w.get("constructively_witnessed") for w in witness_refs)
            ordered = []
            blockers: List[str] = []
            team_only = False
            phase_constrained = str(goal.phase_scope).strip().lower() not in {"", "all"}
            for rid, w in zip(needed_rules, witness_refs):
                if w:
                    ordered.extend(w.get("ordered_path", []))
                    if w.get("witness_type") == "team-only":
                        team_only = True
                    if w.get("phase_constrained"):
                        phase_constrained = True
                    blockers.extend(w.get("blockers", []))
                else:
                    blockers.append(f"missing_rule_witness:{rid}")
            if constructively_witnessed:
                ordered.append(f"ground_goal:{goal.goal_id}")
            out[f"goal:{goal.goal_id}"] = {
                "closure_reachable": closure_reachable,
                "constructively_witnessed": constructively_witnessed,
                "witness_type": "team-only" if team_only else ("individual" if constructively_witnessed else "unwitnessed"),
                "phase_constrained": phase_constrained,
                "phase_scope": goal.phase_scope,
                "communication_required": team_only,
                "derivation_depth": max([w.get("derivation_depth", 0) for w in witness_refs if w] or [0]),
                "upstream_dependency_count": sum(w.get("upstream_dependency_count", 0) for w in witness_refs if w),
                "shared_or_team_only_dependencies": sorted({d for w in witness_refs if w for d in w.get("shared_or_team_only_dependencies", [])}),
                "brittle_chain": any(w.get("brittle_chain") for w in witness_refs if w),
                "ordered_path": list(dict.fromkeys(ordered)),
                "blockers": sorted(set(blockers)),
            }

        for method in task_model.plan_methods.values():
            if not method.enabled:
                continue
            target_id = f"method:{method.method_id}"
            req_rules = sorted(method.required_rules)
            phase_scope = method.phase_scope or "all"
            closure_reachable = (
                set(req_rules).issubset(reachable_rules_team)
                and set(method.required_knowledge).issubset(team_state.reachable)
                and set(method.required_information).issubset(team_state.reachable)
                and set(method.required_data).issubset(team_state.reachable)
            )
            ordered: List[str] = []
            blockers: List[str] = []
            team_only = False
            depth = 0
            deps: Set[str] = set()

            for rid in req_rules:
                w = out.get(f"rule:{rid}")
                if not w:
                    blockers.append(f"missing_rule_witness:{rid}")
                    continue
                ordered.extend(w.get("ordered_path", []))
                depth = max(depth, int(w.get("derivation_depth", 0)))
                deps |= set(w.get("shared_or_team_only_dependencies", []))
                if w.get("witness_type") == "team-only":
                    team_only = True
                blockers.extend(w.get("blockers", []))

            for eid in sorted(set(method.required_knowledge + method.required_information + method.required_data)):
                ew = self._element_witness(task_model, eid, team_state.reachable, phase_scope)
                if ew["ok"]:
                    ordered.extend(ew["steps"])
                    depth = max(depth, int(ew["depth"]))
                    deps |= set(ew["deps"])
                    role_set = self._roles_for_element(role_states, eid)
                    if len(role_set) == 0:
                        if eid in team_state.reachable:
                            team_only = True
                        else:
                            blockers.append(f"no_role_access:{eid}")
                    elif len(role_set) > 1:
                        team_only = True
                else:
                    blockers.extend(ew["blockers"])

            constructively_witnessed = closure_reachable and not blockers
            if team_only and constructively_witnessed:
                ordered.append("communicate_integrate:cross_role_dependency")
            if constructively_witnessed:
                ordered.append(f"ground_plan_method:{method.method_id}")

            out[target_id] = {
                "closure_reachable": closure_reachable,
                "constructively_witnessed": constructively_witnessed,
                "witness_type": "team-only" if team_only else ("individual" if constructively_witnessed else "unwitnessed"),
                "phase_constrained": str(phase_scope).strip().lower() not in {"", "all"},
                "phase_scope": phase_scope,
                "communication_required": team_only,
                "derivation_depth": depth,
                "upstream_dependency_count": len(deps),
                "shared_or_team_only_dependencies": sorted(deps),
                "brittle_chain": depth >= 4 or len(deps) >= 8,
                "ordered_path": list(dict.fromkeys(ordered)),
                "blockers": sorted(set(blockers)),
            }

        return out

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
                normalized = normalize_rule_token(token)
                if normalized and normalized not in task_model.rules:
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
