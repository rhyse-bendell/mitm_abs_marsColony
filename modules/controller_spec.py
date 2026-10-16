"""Per-agent controller specification (controller-neutral configuration; no inference code lives here).

Controller assignment is PER AGENT: every Mars role resolves independently to its own ControllerSpec. A spec names the controller
implementation (`controller_type`) and that controller's own settings (`settings`); the agent's experimental profile (gestalts and the
seven constituents) is configured separately and never folded into controller identity.

Only `procedural_baseline` is enabled. `local_ollama` is a declared placeholder: its settings have a defined place (model, temperature,
controller_generation_seed, context/output limits, timeout, prompt version, inference parameters) so configurations, conditions and
manifests can already describe mixed teams, but instantiating it raises ControllerNotEnabledError. No network or provider code is
imported or invoked from this module.

Randomness: the world/simulation seed is `SimulationState.seed`. `controller_generation_seed` is a separate, optional, per-agent
setting reserved for future generative controllers; it is never derived from, and never consumes, the world RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

SCHEMA = "controller_assignment/1"
PROCEDURAL_BASELINE = "procedural_baseline"
LOCAL_OLLAMA = "local_ollama"

# Reserved settings for generative controllers (all optional; unset = controller default). Procedural agents carry none.
GENERATIVE_SETTING_FIELDS = (
    "model",  # model name/tag, e.g. "qwen2.5:7b"
    "temperature",
    "controller_generation_seed",  # per-agent generation seed; independent of the world seed
    "max_context_tokens",
    "max_output_tokens",
    "timeout_s",
    "prompt_version",
    "inference_parameters",  # free-form dict of additional inference options (top_p, num_predict, ...)
)


class ControllerNotEnabledError(RuntimeError):
    """A declared-but-unimplemented controller was asked to run."""


class ControllerConfigError(ValueError):
    """An invalid controller specification or assignment."""


@dataclass(frozen=True)
class ControllerType:
    controller_type: str
    label: str
    enabled: bool
    backend: Optional[str]  # legacy brain-backend id the runtime resolves this controller through
    implementation: str
    implementation_version: Optional[str]
    settings_fields: tuple = ()


CONTROLLER_TYPES: Dict[str, ControllerType] = {
    PROCEDURAL_BASELINE: ControllerType(PROCEDURAL_BASELINE, "Procedural Baseline", True, "rule_brain", "modules.procedural_baseline_pilot.ProceduralBaselinePilotAdapter", "1", ()),
    LOCAL_OLLAMA: ControllerType(LOCAL_OLLAMA, "Local Ollama (not yet available)", False, None, "(not implemented)", None, GENERATIVE_SETTING_FIELDS),
}


def enabled_controller_types() -> list[str]:
    return [k for k, v in CONTROLLER_TYPES.items() if v.enabled]


def controller_label(controller_type: str) -> str:
    return CONTROLLER_TYPES[controller_type].label


def controller_type_from_label(label: str) -> str:
    for key, ct in CONTROLLER_TYPES.items():
        if label in {ct.label, key}:
            return key
    raise ControllerConfigError(f"unknown controller label {label!r}")


@dataclass(frozen=True)
class ControllerSpec:
    controller_type: str = PROCEDURAL_BASELINE
    settings: Mapping[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------------------------------------------------
    @staticmethod
    def from_value(value: Any) -> "ControllerSpec":
        if isinstance(value, ControllerSpec):
            return value
        if value is None:
            return ControllerSpec()
        if isinstance(value, str):
            return ControllerSpec(controller_type=controller_type_from_label(value.strip()), settings={}).validated()
        if isinstance(value, Mapping):
            allowed = {"controller_type", "settings", "experimental_profile"}
            extra = set(value) - allowed
            if extra:
                raise ControllerConfigError(f"unknown controller keys {sorted(extra)}; allowed {sorted(allowed)}")
            ctype = controller_type_from_label(str(value.get("controller_type", PROCEDURAL_BASELINE)))
            return ControllerSpec(controller_type=ctype, settings=dict(value.get("settings") or {})).validated()
        raise ControllerConfigError(f"cannot read a controller specification from {type(value).__name__}")

    def validated(self) -> "ControllerSpec":
        ct = CONTROLLER_TYPES.get(self.controller_type)
        if ct is None:
            raise ControllerConfigError(f"unknown controller_type {self.controller_type!r}; known {sorted(CONTROLLER_TYPES)}")
        unknown = set(self.settings) - set(ct.settings_fields)
        if unknown:
            raise ControllerConfigError(f"controller {self.controller_type!r} accepts settings {list(ct.settings_fields)}; got unknown {sorted(unknown)}")
        return self

    # -- views ----------------------------------------------------------------------------------------------------------
    @property
    def type_info(self) -> ControllerType:
        return CONTROLLER_TYPES[self.controller_type]

    @property
    def enabled(self) -> bool:
        return self.type_info.enabled

    def require_enabled(self) -> "ControllerSpec":
        if not self.enabled:
            raise ControllerNotEnabledError(f"controller {self.controller_type!r} is declared but not implemented; enabled controllers: {enabled_controller_types()}")
        return self

    def backend(self) -> str:
        """Legacy brain-backend id the existing per-agent runtime resolves for this controller (enabled controllers only)."""
        self.require_enabled()
        return str(self.type_info.backend)

    def full_settings(self) -> Dict[str, Any]:
        """Every settings field the controller type defines (unset ones as None) so a manifest shows the reserved place."""
        out: Dict[str, Any] = {k: None for k in self.type_info.settings_fields}
        out.update(dict(self.settings))
        return out

    def to_dict(self) -> Dict[str, Any]:
        ct = self.type_info
        return {
            "controller_type": self.controller_type,
            "settings": self.full_settings(),
            "enabled": ct.enabled,
            "implementation": ct.implementation,
            "implementation_version": ct.implementation_version,
        }

    def identity(self) -> str:
        """Stable short token for condition identity (settings included so two Ollama models differ)."""
        if not self.settings:
            return self.controller_type
        body = ",".join(f"{k}={self.settings[k]}" for k in sorted(self.settings))
        return f"{self.controller_type}[{body}]"


def resolve_controller_assignments(agent_keys: Iterable[str], assignments: Optional[Mapping[str, Any]] = None, default: Any = PROCEDURAL_BASELINE) -> Dict[str, ControllerSpec]:
    """One independent ControllerSpec per agent key (role). Unlisted agents get `default`; an assignment naming an unknown agent is an error."""
    keys = [str(k) for k in agent_keys]
    assignments = dict(assignments or {})
    unknown = sorted(set(assignments) - set(keys))
    if unknown:
        raise ControllerConfigError(f"controller assignment names unknown agents {unknown}; agents are {keys}")
    default_spec = ControllerSpec.from_value(default)
    return {k: (ControllerSpec.from_value(assignments[k]) if k in assignments else default_spec) for k in keys}


def condition_identity(specs: Mapping[str, ControllerSpec]) -> str:
    """Controller composition as part of a condition's identity, e.g. 'Architect=procedural_baseline|Botanist=local_ollama[model=a]|...'."""
    return "|".join(f"{k}={specs[k].identity()}" for k in sorted(specs))


def assignments_to_serializable(specs: Mapping[str, ControllerSpec]) -> Dict[str, Dict[str, Any]]:
    return {k: {"controller_type": v.controller_type, "settings": dict(v.settings)} for k, v in specs.items()}


def parse_cli_assignments(pairs: Iterable[str]) -> Dict[str, Any]:
    """`--controller Botanist=local_ollama --controller Engineer=procedural_baseline` -> {"Botanist": ..., "Engineer": ...}."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ControllerConfigError(f"expected Role=controller_type, got {pair!r}")
        role, ctype = pair.split("=", 1)
        out[role.strip()] = ctype.strip()
    return out
