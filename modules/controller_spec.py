"""Per-agent controller specification (controller-neutral configuration; no inference code lives here).

Controller assignment is PER AGENT: every Mars role resolves independently to its own ControllerSpec. A spec names the controller
implementation (`controller_type`) and that controller's own settings (`settings`); the agent's experimental profile (gestalts and the
seven constituents) is configured separately and never folded into controller identity.

ControllerSpec is the ONLY source of controller identity. The legacy `brain_backend` strings that used to instantiate an Ollama provider
independently are retired: `LEGACY_OLLAMA_BACKENDS` are routed through a `local_ollama` ControllerSpec by the simulation (or rejected),
and `create_brain_provider` no longer builds an Ollama provider.

Controllers:
- `procedural_baseline`: the Procedural Baseline Pilot (no settings).
- `local_ollama`: a LOCAL Ollama model on a loopback endpoint (see modules/ollama_pilot.py). Settings below. No cloud provider exists.

Randomness: the world/simulation seed is `SimulationState.seed`. `controller_generation_seed` is a separate, optional, per-agent setting for
generative controllers; it is never derived from, and never consumes, the world RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

SCHEMA = "controller_assignment/1"
PROCEDURAL_BASELINE = "procedural_baseline"
LOCAL_OLLAMA = "local_ollama"
HUMAN_PARTICIPANT = "human_participant"

# Execution modes. `simulation` (default) is the original headless/GUI stepping: simulated time advances only when the driver steps it.
# `realtime_testbed` paces simulated time against the wall clock; it is REQUIRED when any agent is a human participant.
EXECUTION_MODE_SIMULATION = "simulation"
EXECUTION_MODE_REALTIME = "realtime_testbed"
EXECUTION_MODES = (EXECUTION_MODE_SIMULATION, EXECUTION_MODE_REALTIME)

# Legacy brain_backend aliases that used to construct an Ollama provider on their own. They are no longer a controller-selection path.
LEGACY_OLLAMA_BACKENDS = ("local_http", "openai_compatible_local", "ollama_local", "ollama")

GENERATIVE_SETTING_FIELDS = (
    "model",  # installed local model name/tag, e.g. "qwen3.5:9b" (required; never pulled automatically)
    "base_url",  # loopback Ollama endpoint
    "temperature",
    "controller_generation_seed",  # per-agent generation seed (Ollama options.seed); independent of the world seed
    "max_context_tokens",  # Ollama num_ctx
    "max_output_tokens",  # Ollama num_predict
    "prompt_budget_tokens",  # budget for the prompt itself; None = max_context - max_output - reserve
    "timeout_s",  # wall-clock per inference request
    "prompt_version",
    "max_retries",  # bounded retries after an unparsable/illegal reply
    "failure_policy",  # what happens when retries are exhausted: "stop_run" (default) | "idle_and_log" (explicit, logged; never another controller)
    "think",  # request model 'thinking' output (default False; no hidden chain-of-thought is requested)
    "warmup",  # load the model before simulated time starts (default True; loads an installed model only, never downloads)
    "keep_alive",  # how long Ollama keeps the model resident between calls (default "30m")
    "inference_parameters",  # additional SAFE Ollama options only (see SAFE_INFERENCE_PARAMETERS)
)

# A human participant has no inference settings. `participant_id` is an optional pseudonymous label recorded in traces/manifests.
HUMAN_SETTING_FIELDS = ("participant_id",)

LOCAL_OLLAMA_DEFAULTS: Dict[str, Any] = {
    "model": None, "base_url": "http://127.0.0.1:11434", "temperature": 0.2, "controller_generation_seed": None, "max_context_tokens": 8192,
    "max_output_tokens": 400, "prompt_budget_tokens": None, "timeout_s": 120.0, "prompt_version": "canonical_choice_v2", "max_retries": 2,
    "failure_policy": "stop_run", "think": False, "warmup": True, "keep_alive": "30m", "inference_parameters": {},
}
FAILURE_POLICIES = ("stop_run", "idle_and_log")
SAFE_INFERENCE_PARAMETERS = ("top_p", "top_k", "min_p", "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty", "num_keep", "mirostat", "mirostat_tau", "mirostat_eta", "typical_p")


class ControllerNotEnabledError(RuntimeError):
    """A declared-but-unimplemented controller was asked to run."""


class ControllerConfigError(ValueError):
    """An invalid controller specification or assignment."""


@dataclass(frozen=True)
class ControllerType:
    controller_type: str
    label: str
    enabled: bool
    backend: Optional[str]  # runtime backend id this controller resolves through (never a free provider choice)
    implementation: str
    implementation_version: Optional[str]
    settings_fields: tuple = ()


CONTROLLER_TYPES: Dict[str, ControllerType] = {
    PROCEDURAL_BASELINE: ControllerType(PROCEDURAL_BASELINE, "Procedural Baseline", True, "rule_brain", "modules.procedural_baseline_pilot.ProceduralBaselinePilotAdapter", "1", ()),
    LOCAL_OLLAMA: ControllerType(LOCAL_OLLAMA, "Local Ollama", True, "local_ollama_controller", "modules.ollama_pilot.OllamaPilotAdapter", "1", GENERATIVE_SETTING_FIELDS),
    HUMAN_PARTICIPANT: ControllerType(HUMAN_PARTICIPANT, "Human Participant", True, "human_participant_controller", "modules.human_participant.HumanParticipantPilotAdapter", "1", HUMAN_SETTING_FIELDS),
}


def resolve_execution_mode(requested: Optional[str], specs: Mapping[str, "ControllerSpec"]) -> str:
    """The execution mode for a controller composition. A human participant requires `realtime_testbed` (the mission clock cannot wait for a
    person); asking for `simulation` with a human present is a configuration error, never a silent switch. No human: default `simulation`."""
    has_human = any(getattr(s, "controller_type", None) == HUMAN_PARTICIPANT for s in specs.values())
    if requested is not None and requested not in EXECUTION_MODES:
        raise ControllerConfigError(f"unknown execution_mode {requested!r}; known {list(EXECUTION_MODES)}")
    if has_human:
        if requested == EXECUTION_MODE_SIMULATION:
            raise ControllerConfigError("a human_participant controller requires execution_mode='realtime_testbed' (the simulated clock cannot be stepped while a person decides)")
        return EXECUTION_MODE_REALTIME
    return requested or EXECUTION_MODE_SIMULATION


def enabled_controller_types() -> list[str]:
    return [k for k, v in CONTROLLER_TYPES.items() if v.enabled]


def controller_label(controller_type: str) -> str:
    return CONTROLLER_TYPES[controller_type].label


def controller_type_from_label(label: str) -> str:
    for key, ct in CONTROLLER_TYPES.items():
        if label in {ct.label, key}:
            return key
    raise ControllerConfigError(f"unknown controller label {label!r}")


def _validate_local_ollama_settings(settings: Mapping[str, Any]) -> None:
    from modules.ollama_client import NonLoopbackEndpointError, validate_loopback_endpoint

    if settings.get("base_url") is not None:
        try:
            validate_loopback_endpoint(str(settings["base_url"]))
        except NonLoopbackEndpointError as exc:
            raise ControllerConfigError(f"local_ollama endpoint rejected: {exc}") from exc

    def num(name, lo=None, hi=None, kind=float):
        v = settings.get(name)
        if v is None:
            return
        try:
            v = kind(v)
        except (TypeError, ValueError):
            raise ControllerConfigError(f"local_ollama setting {name!r} must be a {kind.__name__}, got {settings.get(name)!r}") from None
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ControllerConfigError(f"local_ollama setting {name!r}={v} out of range [{lo}, {hi}]")

    num("temperature", 0.0, 2.0)
    num("controller_generation_seed", None, None, int)
    num("max_context_tokens", 512, 1_048_576, int)
    num("max_output_tokens", 16, 32768, int)
    num("prompt_budget_tokens", 256, 1_048_576, int)
    num("timeout_s", 1.0, 3600.0)
    num("max_retries", 0, 5, int)
    if settings.get("failure_policy") is not None and settings["failure_policy"] not in FAILURE_POLICIES:
        raise ControllerConfigError(f"failure_policy must be one of {FAILURE_POLICIES}")
    params = settings.get("inference_parameters")
    if params is not None:
        if not isinstance(params, Mapping):
            raise ControllerConfigError("inference_parameters must be an object")
        unsafe = sorted(set(params) - set(SAFE_INFERENCE_PARAMETERS))
        if unsafe:
            raise ControllerConfigError(f"inference_parameters {unsafe} are not allowed; safe options: {list(SAFE_INFERENCE_PARAMETERS)} (model, seed, temperature, num_ctx, num_predict have their own fields)")


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
        if self.controller_type == LOCAL_OLLAMA:
            _validate_local_ollama_settings(self.settings)
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

    def require_ready(self) -> "ControllerSpec":
        """Settings sufficient to instantiate (a local_ollama controller needs an explicit model)."""
        self.require_enabled()
        if self.controller_type == LOCAL_OLLAMA and not str(self.settings.get("model") or "").strip():
            raise ControllerConfigError("local_ollama controller requires an explicit installed model (settings.model)")
        return self

    def backend(self) -> str:
        """Runtime backend id this controller resolves through (enabled controllers only)."""
        self.require_enabled()
        return str(self.type_info.backend)

    def full_settings(self) -> Dict[str, Any]:
        """Every settings field the controller type defines, defaults filled in, so a manifest records the exact resolved configuration."""
        if self.controller_type == LOCAL_OLLAMA:
            out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in LOCAL_OLLAMA_DEFAULTS.items()}
        else:
            out = {k: None for k in self.type_info.settings_fields}
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
    """`--controller Botanist=local_ollama:qwen3.5:9b` -> {"Botanist": {"controller_type": "local_ollama", "settings": {"model": "qwen3.5:9b"}}};
    `--controller Engineer=procedural_baseline` -> {"Engineer": "procedural_baseline"}."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ControllerConfigError(f"expected Role=controller_type[:model], got {pair!r}")
        role, rest = pair.split("=", 1)
        ctype, _, model = rest.partition(":")
        ctype = ctype.strip()
        if ctype == LOCAL_OLLAMA and model.strip():
            out[role.strip()] = {"controller_type": LOCAL_OLLAMA, "settings": {"model": model.strip()}}
        else:
            out[role.strip()] = ctype
    return out
