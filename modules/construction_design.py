"""Structured construction design: what a participant chooses when authoring a structure, and the authoritative evaluation of that choice.

Generalised engine layer. The Mars specifics (which properties exist, their options, which values satisfy the task rules and which role-brief facts
justify them) live in `config/tasks/<task_id>/construction_design.json`. Nothing here is a Lego/voxel model: a design is a small mapping
`{property_id: option_value}` covering exactly the attributes the task's validity rules judge (plus free choices the validator does not judge).

Visibility contract
* `public_schema` (property + option labels) is what a participant may see. It carries no `valid` list and no source ids.
* `evaluate` returns the authoritative result `{valid, violations}`. It is hidden world truth: it is stored on the project for the construction validator
  and research logs and must never be sent to a participant.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "construction_design/1"


@lru_cache(maxsize=8)
def _load(task_id: str) -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "tasks" / str(task_id) / "construction_design.json"
    if not path.is_file():
        return {"structures": {}}
    return json.loads(path.read_text(encoding="utf-8"))


class DesignSchema:
    def __init__(self, task_id: str = "mars_colony"):
        self.task_id = str(task_id)
        self._structures: Dict[str, Any] = dict(_load(self.task_id).get("structures") or {})

    def has(self, structure_type: str) -> bool:
        return str(structure_type or "").lower() in self._structures

    def properties(self, structure_type: str) -> List[Dict[str, Any]]:
        return list((self._structures.get(str(structure_type or "").lower()) or {}).get("properties") or [])

    def public_schema(self, structure_type: str) -> Optional[Dict[str, Any]]:
        """Participant-visible form definition (labels only). None if this structure type has no designable properties."""
        entry = self._structures.get(str(structure_type or "").lower())
        if not entry:
            return None
        return {
            "structure_type": str(structure_type).lower(),
            "title": entry.get("title") or "Design",
            "properties": [
                {"id": p["id"], "label": p["label"], "options": [{"value": o["value"], "label": o["label"]} for o in p["options"]]}
                for p in entry.get("properties", [])
            ],
        }

    def validate_choice(self, structure_type: str, design: Any, *, complete: bool) -> List[str]:
        """Structural check of a submitted design (known properties, offered option values; every property chosen when `complete`). -> error tokens."""
        if not self.has(structure_type):
            return ["design_not_supported_for_structure"] if design else []
        if not isinstance(design, Mapping):
            return ["design_not_a_mapping"]
        props = {p["id"]: p for p in self.properties(structure_type)}
        errors: List[str] = []
        for key, value in design.items():
            prop = props.get(str(key))
            if prop is None:
                errors.append(f"design_unknown_property:{key}")
            elif str(value) not in {o["value"] for o in prop["options"]}:
                errors.append(f"design_invalid_option:{key}")
        if complete:
            errors.extend(f"design_incomplete:{pid}" for pid in props if pid not in design)
        return errors

    def evaluate(self, structure_type: str, design: Mapping[str, Any]) -> Dict[str, Any]:
        """AUTHORITATIVE, HIDDEN. {valid, violations:[{property, value, sources}]}. Properties without a `valid` list are not judged."""
        violations = []
        for prop in self.properties(structure_type):
            valid = prop.get("valid")
            if valid is None:
                continue
            value = str((design or {}).get(prop["id"]))
            if value not in set(valid):
                violations.append({"property": prop["id"], "value": value, "sources": list(prop.get("sources") or [])})
        return {"valid": not violations, "violations": violations}

    def labelled(self, structure_type: str, design: Mapping[str, Any]) -> Dict[str, str]:
        """{property label: option label} for display of a participant's OWN choices (physical appearance); no validity information."""
        out = {}
        for prop in self.properties(structure_type):
            if prop["id"] in (design or {}):
                label = next((o["label"] for o in prop["options"] if o["value"] == str(design[prop["id"]])), str(design[prop["id"]]))
                out[prop["label"]] = label
        return out


def get_schema(task_id: str = "mars_colony") -> DesignSchema:
    return DesignSchema(task_id)
