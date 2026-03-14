from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Dict, Iterable, List, Optional


class TaskModelError(Exception):
    """Raised when a task package is missing required files or has invalid rows."""


REQUIRED_TASK_FILES = {
    "task_manifest": "task_manifest.json",
    "task_sources": "task_sources.csv",
    "source_contents": "source_contents.csv",
    "dik_elements": "dik_elements.csv",
    "dik_derivations": "dik_derivations.csv",
    "rule_definitions": "rule_definitions.csv",
    "goal_definitions": "goal_definitions.csv",
    "plan_methods": "plan_methods.csv",
    "artifact_definitions": "artifact_definitions.csv",
    "environment_objects": "environment_objects.csv",
    "zones": "zones.csv",
    "interaction_targets": "interaction_targets.csv",
    "spawn_points": "spawn_points.csv",
    "resource_nodes": "resource_nodes.csv",
    "phase_definitions": "phase_definitions.csv",
    "role_definitions": "role_definitions.csv",
    "agent_defaults": "agent_defaults.csv",
    "action_availability": "action_availability.csv",
    "action_parameters": "action_parameters.csv",
    "communication_catalog": "communication_catalog.csv",
    "construction_templates": "construction_templates.csv",
}


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: str, default: float = 0.0) -> float:
    return float(value) if str(value).strip() else default


def _split_pipe(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split("|") if v.strip()]


@dataclass(frozen=True)
class TaskSource:
    source_id: str
    source_type: str
    role_scope: str
    access_scope: str
    canonical: bool
    enabled: bool
    description: str
    source_doc: str


@dataclass(frozen=True)
class SourceContent:
    source_id: str
    element_id: str
    element_type: str
    access_mode: str
    enabled: bool
    notes: str


@dataclass(frozen=True)
class DIKElement:
    element_id: str
    element_type: str
    label: str
    description: str
    role_scope: str
    phase_scope: str
    canonical: bool
    enabled: bool
    canonical_source: str
    notes: str


@dataclass(frozen=True)
class DIKDerivation:
    derivation_id: str
    output_element_id: str
    output_type: str
    derivation_kind: str
    required_inputs: List[str]
    optional_inputs: List[str]
    min_required_count: int
    enabled: bool
    notes: str


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    label: str
    description: str
    role_scope: str
    phase_scope: str
    required_knowledge: List[str]
    required_information: List[str]
    enabled: bool
    notes: str


@dataclass(frozen=True)
class GoalDefinition:
    goal_id: str
    goal_level: str
    label: str
    description: str
    phase_scope: str
    success_conditions: str
    prerequisite_rules: List[str]
    enabled: bool
    notes: str


@dataclass(frozen=True)
class PlanMethod:
    method_id: str
    goal_id: str
    label: str
    description: str
    phase_scope: str
    required_rules: List[str]
    required_knowledge: List[str]
    required_information: List[str]
    required_data: List[str]
    candidate_steps: List[str]
    completion_conditions: str
    enabled: bool
    notes: str


@dataclass(frozen=True)
class ArtifactDefinition:
    artifact_type: str
    label: str
    description: str
    represents: List[str]
    validation_enabled: bool
    consultable: bool
    revisable: bool
    enabled: bool
    notes: str


@dataclass(frozen=True)
class EnvironmentObject:
    object_id: str
    object_type: str
    x: float
    y: float
    width: float
    height: float
    radius: float
    end_x: float
    end_y: float
    label: str
    passable: bool
    access_radius: float
    orientation: str
    role_restriction: str
    enabled: bool


@dataclass(frozen=True)
class ZoneDefinition:
    zone_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    default_zone: bool
    enabled: bool


@dataclass(frozen=True)
class InteractionTarget:
    target_id: str
    kind: str
    zone_id: str
    object_id: str
    role_scope: List[str]
    enabled: bool


@dataclass(frozen=True)
class SpawnPoint:
    spawn_id: str
    role_id: str
    x: float
    y: float
    priority: int
    enabled: bool


@dataclass(frozen=True)
class ResourceNode:
    node_id: str
    zone_id: str
    resource_type: str
    quantity: int
    x: float
    y: float
    transport_time_scale: float
    enabled: bool


@dataclass(frozen=True)
class PhaseDefinition:
    phase_id: str
    name: str
    duration_minutes: float
    colonist_manifest: Dict[str, int]
    unlocks: List[str]
    required_structures: Dict[str, Dict[str, int]]
    description: str
    enabled: bool


@dataclass(frozen=True)
class RoleDefinition:
    role_id: str
    label: str
    default_active: bool
    spawn_id: str
    source_scope: List[str]
    enabled: bool


@dataclass(frozen=True)
class AgentDefault:
    role_id: str
    agent_name: str
    teamwork_potential: float
    taskwork_potential: float
    mechanism_overrides: Dict[str, float]
    source_access_override: List[str]
    planner_config: Dict[str, object]
    enabled: bool


@dataclass(frozen=True)
class ActionAvailability:
    action_id: str
    enabled: bool
    role_scope: List[str]
    target_kinds: List[str]


@dataclass(frozen=True)
class ActionParameter:
    action_id: str
    duration_s: float
    metadata: Dict[str, str]


@dataclass(frozen=True)
class CommunicationDefinition:
    code: str
    label: str
    enabled: bool


@dataclass(frozen=True)
class ConstructionTemplate:
    project_id: str
    name: str
    structure_type: str
    target_id: str
    location_x: float
    location_y: float
    required_resources: Dict[str, int]
    expected_rules: List[str]
    artifact_type: str
    enabled: bool


@dataclass
class TaskModel:
    task_id: str
    base_path: Path
    manifest: Dict[str, object]
    sources: Dict[str, TaskSource]
    source_contents: List[SourceContent]
    dik_elements: Dict[str, DIKElement]
    derivations: Dict[str, DIKDerivation]
    rules: Dict[str, RuleDefinition]
    goals: Dict[str, GoalDefinition]
    plan_methods: Dict[str, PlanMethod]
    artifacts: Dict[str, ArtifactDefinition]
    environment_objects: Dict[str, EnvironmentObject]
    zones: Dict[str, ZoneDefinition]
    interaction_targets: Dict[str, InteractionTarget]
    spawn_points: List[SpawnPoint]
    resource_nodes: Dict[str, ResourceNode]
    phases: List[PhaseDefinition]
    roles: Dict[str, RoleDefinition]
    agent_defaults: List[AgentDefault]
    action_availability: Dict[str, ActionAvailability]
    action_parameters: Dict[str, ActionParameter]
    communication_catalog: Dict[str, CommunicationDefinition]
    construction_templates: Dict[str, ConstructionTemplate]

    def enabled_sources(self) -> Iterable[TaskSource]:
        return (s for s in self.sources.values() if s.enabled)

    def elements_for_source(self, source_id: str, *, element_type: Optional[str] = None) -> List[DIKElement]:
        rows = [
            r
            for r in self.source_contents
            if r.enabled and r.source_id == source_id and (element_type is None or r.element_type == element_type)
        ]
        out: List[DIKElement] = []
        for row in rows:
            element = self.dik_elements.get(row.element_id)
            if element and element.enabled:
                out.append(element)
        return out

    def source_ids_for_role(self, role: str) -> List[str]:
        role_normalized = (role or "").strip().lower()
        allowed = []
        for source in self.enabled_sources():
            scope = source.access_scope.strip().lower()
            if scope in {"all", "team"} or scope == role_normalized:
                allowed.append(source.source_id)
        return allowed

    def enabled_actions_for_role(self, role: str) -> List[str]:
        role_key = (role or "").strip().lower()
        enabled = []
        for action_id, rule in self.action_availability.items():
            if not rule.enabled:
                continue
            if not rule.role_scope or "all" in {s.lower() for s in rule.role_scope}:
                enabled.append(action_id)
                continue
            if role_key and role_key in {s.lower() for s in rule.role_scope}:
                enabled.append(action_id)
        return enabled


class TaskModelLoader:
    def __init__(self, config_root: str | Path = "config/tasks"):
        self.config_root = Path(config_root)

    def load(self, task_id: str = "mars_colony") -> TaskModel:
        task_path = self.config_root / task_id
        if not task_path.exists():
            raise TaskModelError(f"Task package not found: {task_path}")

        missing = [fname for fname in REQUIRED_TASK_FILES.values() if not (task_path / fname).exists()]
        if missing:
            raise TaskModelError(f"Task package '{task_id}' missing required files: {', '.join(sorted(missing))}")

        return TaskModel(
            task_id=task_id,
            base_path=task_path,
            manifest=self._load_manifest(task_path / REQUIRED_TASK_FILES["task_manifest"]),
            sources=self._load_sources(task_path / REQUIRED_TASK_FILES["task_sources"]),
            source_contents=self._load_source_contents(task_path / REQUIRED_TASK_FILES["source_contents"]),
            dik_elements=self._load_dik_elements(task_path / REQUIRED_TASK_FILES["dik_elements"]),
            derivations=self._load_derivations(task_path / REQUIRED_TASK_FILES["dik_derivations"]),
            rules=self._load_rules(task_path / REQUIRED_TASK_FILES["rule_definitions"]),
            goals=self._load_goals(task_path / REQUIRED_TASK_FILES["goal_definitions"]),
            plan_methods=self._load_plan_methods(task_path / REQUIRED_TASK_FILES["plan_methods"]),
            artifacts=self._load_artifacts(task_path / REQUIRED_TASK_FILES["artifact_definitions"]),
            environment_objects=self._load_environment_objects(task_path / REQUIRED_TASK_FILES["environment_objects"]),
            zones=self._load_zones(task_path / REQUIRED_TASK_FILES["zones"]),
            interaction_targets=self._load_interaction_targets(task_path / REQUIRED_TASK_FILES["interaction_targets"]),
            spawn_points=self._load_spawn_points(task_path / REQUIRED_TASK_FILES["spawn_points"]),
            resource_nodes=self._load_resource_nodes(task_path / REQUIRED_TASK_FILES["resource_nodes"]),
            phases=self._load_phase_definitions(task_path / REQUIRED_TASK_FILES["phase_definitions"]),
            roles=self._load_role_definitions(task_path / REQUIRED_TASK_FILES["role_definitions"]),
            agent_defaults=self._load_agent_defaults(task_path / REQUIRED_TASK_FILES["agent_defaults"]),
            action_availability=self._load_action_availability(task_path / REQUIRED_TASK_FILES["action_availability"]),
            action_parameters=self._load_action_parameters(task_path / REQUIRED_TASK_FILES["action_parameters"]),
            communication_catalog=self._load_communication_catalog(task_path / REQUIRED_TASK_FILES["communication_catalog"]),
            construction_templates=self._load_construction_templates(task_path / REQUIRED_TASK_FILES["construction_templates"]),
        )

    @staticmethod
    def _rows(path: Path) -> List[dict]:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
            if any(not fn for fn in fieldnames):
                raise TaskModelError(f"Invalid header in {path}")
            rows = []
            for raw in reader:
                if not raw:
                    continue
                if raw.get(fieldnames[0], "").strip().lower() == fieldnames[0].strip().lower():
                    continue
                rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
            return rows

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise TaskModelError(f"task_manifest.json must contain an object: {path}")
        return payload

    def _load_sources(self, path: Path) -> Dict[str, TaskSource]:
        out = {}
        for row in self._rows(path):
            source = TaskSource(
                source_id=row["source_id"],
                source_type=row["source_type"],
                role_scope=row["role_scope"],
                access_scope=row["access_scope"],
                canonical=_parse_bool(row["canonical"]),
                enabled=_parse_bool(row["enabled"]),
                description=row["description"],
                source_doc=row["source_doc"],
            )
            out[source.source_id] = source
        return out

    def _load_source_contents(self, path: Path) -> List[SourceContent]:
        return [
            SourceContent(
                source_id=row["source_id"],
                element_id=row["element_id"],
                element_type=row["element_type"],
                access_mode=row["access_mode"],
                enabled=_parse_bool(row["enabled"]),
                notes=row["notes"],
            )
            for row in self._rows(path)
        ]

    def _load_dik_elements(self, path: Path) -> Dict[str, DIKElement]:
        out = {}
        for row in self._rows(path):
            element = DIKElement(
                element_id=row["element_id"],
                element_type=row["element_type"],
                label=row["label"],
                description=row["description"],
                role_scope=row["role_scope"],
                phase_scope=row["phase_scope"],
                canonical=_parse_bool(row["canonical"]),
                enabled=_parse_bool(row["enabled"]),
                canonical_source=row["canonical_source"],
                notes=row["notes"],
            )
            out[element.element_id] = element
        return out

    def _load_derivations(self, path: Path) -> Dict[str, DIKDerivation]:
        out = {}
        for row in self._rows(path):
            min_required_raw = row.get("min_required_count", "")
            min_required = int(min_required_raw) if str(min_required_raw).strip() else 0
            derivation = DIKDerivation(
                derivation_id=row["derivation_id"],
                output_element_id=row["output_element_id"],
                output_type=row["output_type"],
                derivation_kind=row["derivation_kind"],
                required_inputs=_split_pipe(row["required_inputs"]),
                optional_inputs=_split_pipe(row["optional_inputs"]),
                min_required_count=min_required,
                enabled=_parse_bool(row["enabled"]),
                notes=row["notes"],
            )
            out[derivation.derivation_id] = derivation
        return out

    def _load_rules(self, path: Path) -> Dict[str, RuleDefinition]:
        out = {}
        for row in self._rows(path):
            rule = RuleDefinition(
                rule_id=row["rule_id"],
                label=row["label"],
                description=row["description"],
                role_scope=row["role_scope"],
                phase_scope=row["phase_scope"],
                required_knowledge=_split_pipe(row["required_knowledge"]),
                required_information=_split_pipe(row["required_information"]),
                enabled=_parse_bool(row["enabled"]),
                notes=row["notes"],
            )
            out[rule.rule_id] = rule
        return out

    def _load_goals(self, path: Path) -> Dict[str, GoalDefinition]:
        out = {}
        for row in self._rows(path):
            goal = GoalDefinition(
                goal_id=row["goal_id"],
                goal_level=row["goal_level"],
                label=row["label"],
                description=row["description"],
                phase_scope=row["phase_scope"],
                success_conditions=row["success_conditions"],
                prerequisite_rules=_split_pipe(row["prerequisite_rules"]),
                enabled=_parse_bool(row["enabled"]),
                notes=row["notes"],
            )
            out[goal.goal_id] = goal
        return out

    def _load_plan_methods(self, path: Path) -> Dict[str, PlanMethod]:
        out = {}
        for row in self._rows(path):
            method = PlanMethod(
                method_id=row["method_id"],
                goal_id=row["goal_id"],
                label=row["label"],
                description=row["description"],
                phase_scope=row["phase_scope"],
                required_rules=_split_pipe(row["required_rules"]),
                required_knowledge=_split_pipe(row["required_knowledge"]),
                required_information=_split_pipe(row["required_information"]),
                required_data=_split_pipe(row["required_data"]),
                candidate_steps=_split_pipe(row["candidate_steps"]),
                completion_conditions=row["completion_conditions"],
                enabled=_parse_bool(row["enabled"]),
                notes=row["notes"],
            )
            out[method.method_id] = method
        return out

    def _load_artifacts(self, path: Path) -> Dict[str, ArtifactDefinition]:
        out = {}
        for row in self._rows(path):
            artifact = ArtifactDefinition(
                artifact_type=row["artifact_type"],
                label=row["label"],
                description=row["description"],
                represents=_split_pipe(row["represents"]),
                validation_enabled=_parse_bool(row["validation_enabled"]),
                consultable=_parse_bool(row["consultable"]),
                revisable=_parse_bool(row["revisable"]),
                enabled=_parse_bool(row["enabled"]),
                notes=row["notes"],
            )
            out[artifact.artifact_type] = artifact
        return out

    def _load_environment_objects(self, path: Path) -> Dict[str, EnvironmentObject]:
        out = {}
        for row in self._rows(path):
            obj = EnvironmentObject(
                object_id=row["object_id"],
                object_type=row["object_type"],
                x=_parse_float(row.get("x", "")),
                y=_parse_float(row.get("y", "")),
                width=_parse_float(row.get("width", "")),
                height=_parse_float(row.get("height", "")),
                radius=_parse_float(row.get("radius", "")),
                end_x=_parse_float(row.get("end_x", "")),
                end_y=_parse_float(row.get("end_y", "")),
                label=row.get("label", ""),
                passable=_parse_bool(row.get("passable", "")),
                access_radius=_parse_float(row.get("access_radius", "")),
                orientation=row.get("orientation", ""),
                role_restriction=row.get("role_restriction", ""),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            out[obj.object_id] = obj
        return out

    def _load_zones(self, path: Path) -> Dict[str, ZoneDefinition]:
        out = {}
        for row in self._rows(path):
            zone = ZoneDefinition(
                zone_id=row["zone_id"],
                x1=_parse_float(row.get("x1", "")),
                y1=_parse_float(row.get("y1", "")),
                x2=_parse_float(row.get("x2", "")),
                y2=_parse_float(row.get("y2", "")),
                default_zone=_parse_bool(row.get("default_zone", "")),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            out[zone.zone_id] = zone
        return out

    def _load_interaction_targets(self, path: Path) -> Dict[str, InteractionTarget]:
        out = {}
        for row in self._rows(path):
            target = InteractionTarget(
                target_id=row["target_id"],
                kind=row["kind"],
                zone_id=row["zone_id"],
                object_id=row.get("object_id", ""),
                role_scope=_split_pipe(row.get("role_scope", "")),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            out[target.target_id] = target
        return out

    def _load_spawn_points(self, path: Path) -> List[SpawnPoint]:
        points = [
            SpawnPoint(
                spawn_id=row["spawn_id"],
                role_id=row["role_id"],
                x=_parse_float(row["x"]),
                y=_parse_float(row["y"]),
                priority=int(row.get("priority", "0") or 0),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            for row in self._rows(path)
        ]
        return sorted(points, key=lambda s: (s.role_id, s.priority, s.spawn_id))

    def _load_resource_nodes(self, path: Path) -> Dict[str, ResourceNode]:
        out = {}
        for row in self._rows(path):
            node = ResourceNode(
                node_id=row["node_id"],
                zone_id=row["zone_id"],
                resource_type=row["resource_type"],
                quantity=int(row.get("quantity", "0") or 0),
                x=_parse_float(row.get("x", "")),
                y=_parse_float(row.get("y", "")),
                transport_time_scale=_parse_float(row.get("transport_time_scale", ""), default=1.0),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            out[node.node_id] = node
        return out

    def _load_phase_definitions(self, path: Path) -> List[PhaseDefinition]:
        phases = []
        for row in self._rows(path):
            phase = PhaseDefinition(
                phase_id=row["phase_id"],
                name=row["name"],
                duration_minutes=_parse_float(row["duration_minutes"]),
                colonist_manifest=json.loads(row.get("colonist_manifest_json", "{}") or "{}"),
                unlocks=_split_pipe(row.get("unlocks", "")),
                required_structures=json.loads(row.get("required_structures_json", "{}") or "{}"),
                description=row.get("description", ""),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            phases.append(phase)
        return [p for p in phases if p.enabled]

    def _load_role_definitions(self, path: Path) -> Dict[str, RoleDefinition]:
        out = {}
        for row in self._rows(path):
            role = RoleDefinition(
                role_id=row["role_id"],
                label=row.get("label", row["role_id"]),
                default_active=_parse_bool(row.get("default_active", "")),
                spawn_id=row.get("spawn_id", ""),
                source_scope=_split_pipe(row.get("source_scope", "")),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            out[role.role_id] = role
        return out

    def _load_agent_defaults(self, path: Path) -> List[AgentDefault]:
        defaults = []
        for row in self._rows(path):
            default = AgentDefault(
                role_id=row["role_id"],
                agent_name=row.get("agent_name", row["role_id"]),
                teamwork_potential=_parse_float(row.get("teamwork_potential", ""), default=0.5),
                taskwork_potential=_parse_float(row.get("taskwork_potential", ""), default=0.5),
                mechanism_overrides=json.loads(row.get("mechanism_overrides_json", "{}") or "{}"),
                source_access_override=_split_pipe(row.get("source_access_override", "")),
                planner_config=json.loads(row.get("planner_config_json", "{}") or "{}"),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            defaults.append(default)
        return [d for d in defaults if d.enabled]

    def _load_action_availability(self, path: Path) -> Dict[str, ActionAvailability]:
        out = {}
        for row in self._rows(path):
            action = ActionAvailability(
                action_id=row["action_id"],
                enabled=_parse_bool(row.get("enabled", "true")),
                role_scope=_split_pipe(row.get("role_scope", "")),
                target_kinds=_split_pipe(row.get("target_kinds", "")),
            )
            out[action.action_id] = action
        return out

    def _load_action_parameters(self, path: Path) -> Dict[str, ActionParameter]:
        out = {}
        for row in self._rows(path):
            params = ActionParameter(
                action_id=row["action_id"],
                duration_s=_parse_float(row.get("duration_s", ""), default=1.0),
                metadata=json.loads(row.get("metadata_json", "{}") or "{}"),
            )
            out[params.action_id] = params
        return out

    def _load_communication_catalog(self, path: Path) -> Dict[str, CommunicationDefinition]:
        out = {}
        for row in self._rows(path):
            entry = CommunicationDefinition(
                code=row["code"],
                label=row.get("label", row["code"]),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            out[entry.code] = entry
        return out

    def _load_construction_templates(self, path: Path) -> Dict[str, ConstructionTemplate]:
        out = {}
        for row in self._rows(path):
            template = ConstructionTemplate(
                project_id=row["project_id"],
                name=row["name"],
                structure_type=row["structure_type"],
                target_id=row["target_id"],
                location_x=_parse_float(row["location_x"]),
                location_y=_parse_float(row["location_y"]),
                required_resources=json.loads(row.get("required_resources_json", "{}") or "{}"),
                expected_rules=_split_pipe(row.get("expected_rules", "")),
                artifact_type=row.get("artifact_type", ""),
                enabled=_parse_bool(row.get("enabled", "true")),
            )
            out[template.project_id] = template
        return out


def load_task_model(task_id: str = "mars_colony", config_root: str | Path = "config/tasks") -> TaskModel:
    return TaskModelLoader(config_root=config_root).load(task_id=task_id)
