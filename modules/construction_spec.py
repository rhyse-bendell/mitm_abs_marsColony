"""Task-package construction specification: the design choices a build step involves and what makes each realized choice valid.

``config/tasks/<task_id>/construction_specs.csv`` lists, per structure type, the build components that involve a specification
choice (a *slot*), the closed set of options every agent is offered, which options satisfy the task requirement (``compliant``,
world truth held by the Environment only), and the DIK elements that state the requirement (``requirement_elements``). A task package
without this file has no specified slots and keeps the earlier construction semantics.

Three things stay separate (see Documentation/research/MITM_contracts.md):
* the INTENDED option, selected by a Pilot (informed by the DIK it holds, or by chance);
* the REALIZED option, which equals the intended one unless an execution slip occurred;
* structural validity, which the Environment derives from the realized options alone.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

SPEC_FILE = "construction_specs.csv"


@dataclass(frozen=True)
class SlotSpec:
    structure_type: str
    component: str
    slot_id: str
    label: str
    options: Tuple[str, ...]
    compliant: FrozenSet[str]
    requirement_elements: Tuple[str, ...]
    source: str

    def is_compliant(self, option: Optional[str]) -> bool:
        return option in self.compliant


def _split(value: str) -> List[str]:
    return [p.strip() for p in str(value or "").split("|") if p.strip()]


def load_construction_specs(task_path) -> Dict[str, Dict[str, SlotSpec]]:
    """{structure_type: {component: SlotSpec}}; empty when the task package defines no specification."""
    path = Path(task_path) / SPEC_FILE if task_path is not None else None
    if path is None or not path.exists():
        return {}
    specs: Dict[str, Dict[str, SlotSpec]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            options = tuple(_split(row["options"]))
            compliant = frozenset(_split(row["compliant"]))
            if not compliant or not compliant.issubset(options):
                raise ValueError(f"{SPEC_FILE}: compliant options of {row['structure_type']}.{row['slot_id']} must be a non-empty subset of options")
            spec = SlotSpec(
                structure_type=row["structure_type"].strip().lower(),
                component=row["component"].strip(),
                slot_id=row["slot_id"].strip(),
                label=row["label"].strip(),
                options=options,
                compliant=compliant,
                requirement_elements=tuple(_split(row["requirement_elements"])),
                source=row["source"].strip(),
            )
            specs.setdefault(spec.structure_type, {})[spec.component] = spec
    return specs


def requirement_held(spec: SlotSpec, held_element_ids) -> bool:
    """True when the holder has any DIK element that states this slot's requirement (own mental model only)."""
    held = set(held_element_ids)
    return any(eid in held for eid in spec.requirement_elements)
