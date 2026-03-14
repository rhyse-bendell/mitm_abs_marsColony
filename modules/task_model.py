from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Dict, Iterable, List, Optional


class TaskModelError(Exception):
    """Raised when a task package is missing required files or has invalid rows."""


REQUIRED_TASK_FILES = {
    "task_sources": "task_sources.csv",
    "source_contents": "source_contents.csv",
    "dik_elements": "dik_elements.csv",
    "dik_derivations": "dik_derivations.csv",
    "rule_definitions": "rule_definitions.csv",
    "goal_definitions": "goal_definitions.csv",
    "plan_methods": "plan_methods.csv",
    "artifact_definitions": "artifact_definitions.csv",
}


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


@dataclass
class TaskModel:
    task_id: str
    base_path: Path
    sources: Dict[str, TaskSource]
    source_contents: List[SourceContent]
    dik_elements: Dict[str, DIKElement]
    derivations: Dict[str, DIKDerivation]
    rules: Dict[str, RuleDefinition]
    goals: Dict[str, GoalDefinition]
    plan_methods: Dict[str, PlanMethod]
    artifacts: Dict[str, ArtifactDefinition]

    def enabled_sources(self) -> Iterable[TaskSource]:
        return (s for s in self.sources.values() if s.enabled)

    def elements_for_source(self, source_id: str, *, element_type: Optional[str] = None) -> List[DIKElement]:
        rows = [
            r for r in self.source_contents
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
            sources=self._load_sources(task_path / REQUIRED_TASK_FILES["task_sources"]),
            source_contents=self._load_source_contents(task_path / REQUIRED_TASK_FILES["source_contents"]),
            dik_elements=self._load_dik_elements(task_path / REQUIRED_TASK_FILES["dik_elements"]),
            derivations=self._load_derivations(task_path / REQUIRED_TASK_FILES["dik_derivations"]),
            rules=self._load_rules(task_path / REQUIRED_TASK_FILES["rule_definitions"]),
            goals=self._load_goals(task_path / REQUIRED_TASK_FILES["goal_definitions"]),
            plan_methods=self._load_plan_methods(task_path / REQUIRED_TASK_FILES["plan_methods"]),
            artifacts=self._load_artifacts(task_path / REQUIRED_TASK_FILES["artifact_definitions"]),
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
                candidate_steps=_split_pipe(row["candidate_steps".strip()]),
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


def load_task_model(task_id: str = "mars_colony", config_root: str | Path = "config/tasks") -> TaskModel:
    return TaskModelLoader(config_root=config_root).load(task_id=task_id)
