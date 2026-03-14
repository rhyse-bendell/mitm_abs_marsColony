from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ConstructDefinition:
    construct_id: str
    label: str
    description: str
    scale_min: float
    scale_max: float
    default_value: float
    construct_group: str
    enabled: bool


@dataclass(frozen=True)
class ConstructMechanismRule:
    construct_id: str
    mechanism_id: str
    effect_weight: float
    transform: str
    intercept: float
    min_output: float
    max_output: float
    enabled: bool


@dataclass(frozen=True)
class MechanismHookRule:
    mechanism_id: str
    hook_type: str
    hook_target: str
    operator: str
    parameter: str
    formula_name: str
    min_effect: float
    max_effect: float
    enabled: bool


TransformFn = Callable[[float], float]
FormulaFn = Callable[[float, float, float], float]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


TRANSFORMS: dict[str, TransformFn] = {
    "linear": lambda v: v,
    "linear_centered": lambda v: (v - 0.5) * 2.0,
}


FORMULAS: dict[str, FormulaFn] = {
    "bounded_add": lambda v, lo, hi: clamp(v, lo, hi),
    "bounded_identity": lambda v, lo, hi: clamp(v, lo, hi),
    "inverse_duration_scale": lambda v, lo, hi: clamp(hi - (hi - lo) * clamp(v, 0.0, 1.0), lo, hi),
    "threshold_shift": lambda v, lo, hi: clamp(((clamp(v, 0.0, 1.0) - 0.5) * 2.0), lo, hi),
}


class ConstructMappingError(ValueError):
    pass


class ConstructMapper:
    def __init__(self, config_dir: str | Path = "config"):
        self.config_dir = Path(config_dir)
        self.constructs: dict[str, ConstructDefinition] = {}
        self.construct_to_mechanism: list[ConstructMechanismRule] = []
        self.mechanism_to_hook: list[MechanismHookRule] = []
        self.validation_issues: list[str] = []
        self._load_all()

    def _parse_bool(self, value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _parse_float(self, row: dict[str, str], key: str, context: str, default: float | None = None) -> float:
        raw = row.get(key, "")
        if raw is None or str(raw).strip() == "":
            if default is not None:
                return default
            raise ConstructMappingError(f"Missing numeric field '{key}' in {context}")
        try:
            return float(raw)
        except ValueError as exc:
            raise ConstructMappingError(f"Invalid numeric field '{key}'='{raw}' in {context}") from exc

    def _load_csv(self, file_name: str) -> list[dict[str, str]]:
        path = self.config_dir / file_name
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _load_all(self) -> None:
        self._load_constructs()
        self._load_construct_to_mechanism()
        self._load_mechanism_to_hook()

    def _load_constructs(self) -> None:
        rows = self._load_csv("constructs.csv")
        for idx, row in enumerate(rows, start=2):
            context = f"constructs.csv row {idx}"
            try:
                scale_min = self._parse_float(row, "scale_min", context)
                scale_max = self._parse_float(row, "scale_max", context)
                default_value = self._parse_float(row, "default_value", context)
            except ConstructMappingError as err:
                self.validation_issues.append(str(err))
                continue
            if scale_max < scale_min:
                self.validation_issues.append(f"Invalid bounds in {context}: scale_max < scale_min")
                continue
            construct = ConstructDefinition(
                construct_id=row["construct_id"].strip(),
                label=row["label"].strip(),
                description=row.get("description", "").strip(),
                scale_min=scale_min,
                scale_max=scale_max,
                default_value=clamp(default_value, scale_min, scale_max),
                construct_group=row.get("construct_group", "").strip(),
                enabled=self._parse_bool(row.get("enabled", "true")),
            )
            self.constructs[construct.construct_id] = construct

    def _load_construct_to_mechanism(self) -> None:
        rows = self._load_csv("construct_to_mechanism.csv")
        for idx, row in enumerate(rows, start=2):
            context = f"construct_to_mechanism.csv row {idx}"
            transform = row["transform"].strip()
            if transform not in TRANSFORMS:
                self.validation_issues.append(f"Unknown transform '{transform}' in {context}")
                continue
            try:
                effect_weight = self._parse_float(row, "effect_weight", context)
                intercept = self._parse_float(row, "intercept", context, default=0.0)
                min_output = self._parse_float(row, "min_output", context, default=0.0)
                max_output = self._parse_float(row, "max_output", context, default=1.0)
            except ConstructMappingError as err:
                self.validation_issues.append(str(err))
                continue
            if max_output < min_output:
                self.validation_issues.append(f"Invalid bounds in {context}: max_output < min_output")
                continue
            rule = ConstructMechanismRule(
                construct_id=row["construct_id"].strip(),
                mechanism_id=row["mechanism_id"].strip(),
                effect_weight=effect_weight,
                transform=transform,
                intercept=intercept,
                min_output=min_output,
                max_output=max_output,
                enabled=self._parse_bool(row.get("enabled", "true")),
            )
            if rule.construct_id not in self.constructs:
                self.validation_issues.append(f"Unknown construct '{rule.construct_id}' in {context}")
                continue
            self.construct_to_mechanism.append(rule)

    def _load_mechanism_to_hook(self) -> None:
        rows = self._load_csv("mechanism_to_hook.csv")
        for idx, row in enumerate(rows, start=2):
            context = f"mechanism_to_hook.csv row {idx}"
            formula_name = row["formula_name"].strip()
            if formula_name not in FORMULAS:
                self.validation_issues.append(f"Unknown formula '{formula_name}' in {context}")
                continue
            try:
                min_effect = self._parse_float(row, "min_effect", context, default=0.0)
                max_effect = self._parse_float(row, "max_effect", context, default=1.0)
            except ConstructMappingError as err:
                self.validation_issues.append(str(err))
                continue
            if max_effect < min_effect:
                self.validation_issues.append(f"Invalid bounds in {context}: max_effect < min_effect")
                continue
            self.mechanism_to_hook.append(
                MechanismHookRule(
                    mechanism_id=row["mechanism_id"].strip(),
                    hook_type=row["hook_type"].strip(),
                    hook_target=row["hook_target"].strip(),
                    operator=row.get("operator", "").strip(),
                    parameter=row.get("parameter", "").strip(),
                    formula_name=formula_name,
                    min_effect=min_effect,
                    max_effect=max_effect,
                    enabled=self._parse_bool(row.get("enabled", "true")),
                )
            )

    def _construct_value(self, construct_id: str, construct_values: dict[str, float]) -> float:
        definition = self.constructs[construct_id]
        raw = float(construct_values.get(construct_id, definition.default_value))
        return clamp(raw, definition.scale_min, definition.scale_max)

    def resolve_mechanisms(
        self,
        construct_values: dict[str, float] | None = None,
        mechanism_overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        construct_values = construct_values or {}
        mechanism_overrides = mechanism_overrides or {}
        resolved: dict[str, float] = {}

        for rule in self.construct_to_mechanism:
            if not rule.enabled:
                continue
            definition = self.constructs.get(rule.construct_id)
            if definition is None or not definition.enabled:
                continue
            construct_value = self._construct_value(rule.construct_id, construct_values)
            transformed = TRANSFORMS[rule.transform](construct_value)
            prior = resolved.get(rule.mechanism_id, 0.5)
            candidate = prior + (transformed * rule.effect_weight) + rule.intercept
            resolved[rule.mechanism_id] = clamp(candidate, rule.min_output, rule.max_output)

        for mechanism_id, override_value in mechanism_overrides.items():
            resolved[mechanism_id] = clamp(float(override_value), 0.0, 1.0)

        return resolved

    def resolve_hooks(self, mechanism_profile: dict[str, float]) -> dict[tuple[str, str, str], float]:
        effects: dict[tuple[str, str, str], float] = {}
        for rule in self.mechanism_to_hook:
            if not rule.enabled:
                continue
            mechanism_value = float(mechanism_profile.get(rule.mechanism_id, 0.5))
            formula = FORMULAS[rule.formula_name]
            effect = formula(mechanism_value, rule.min_effect, rule.max_effect)
            effects[(rule.hook_type, rule.hook_target, rule.parameter)] = effect
        return effects

    def resolve_agent_profile(
        self,
        construct_values: dict[str, float] | None = None,
        mechanism_overrides: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[tuple[str, str, str], float]]:
        construct_values = construct_values or {}
        normalized_constructs: dict[str, float] = {}
        for construct_id, definition in self.constructs.items():
            if not definition.enabled:
                continue
            normalized_constructs[construct_id] = self._construct_value(construct_id, construct_values)

        mechanisms = self.resolve_mechanisms(normalized_constructs, mechanism_overrides)
        hooks = self.resolve_hooks(mechanisms)
        return normalized_constructs, mechanisms, hooks
