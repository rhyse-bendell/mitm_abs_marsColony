"""Compact first-pass experimental profile: gestalt constructs -> seven constituents -> runtime mechanisms.

Resolution happens ONCE per agent at construction. Runtime code never looks at teamwork_potential or taskwork_potential;
it reads the resolved mechanisms. Explicit constituent overrides replace (never add to) gestalt-generated values.
Mappings live in config/experimental/experimental_profile.json and are hypothesized defaults, not calibrated truth.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

CONSTITUENTS = (
    "dik_push_propensity",
    "dik_request_propensity",
    "externalization_propensity",
    "memory_reliability",
    "coordination_propensity",
    "backup_propensity",
    "task_execution_efficiency",
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@lru_cache(maxsize=4)
def _load(path_str: str) -> Dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_experimental_profile_config(config_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    base = Path(config_dir) if config_dir else Path(__file__).resolve().parents[1] / "config" / "experimental"
    path = base / "experimental_profile.json"
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "config" / "experimental" / "experimental_profile.json"
    return _load(str(path))


def resolve_profile_mode(profile_mode: Optional[str] = None, *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve a profile mode name into its residual-mechanism policy (unknown names are an error, never a silent fallback)."""
    cfg = config or load_experimental_profile_config()
    modes = dict((cfg.get("profile_modes") or {}).get("modes") or {})
    name = str(profile_mode or (cfg.get("profile_modes") or {}).get("default") or "broad_legacy")
    if name not in modes:
        raise ValueError(f"Unknown profile mode '{name}'. Known: {sorted(modes)}")
    spec = modes[name]
    residuals = list(spec.get("residual_mechanisms") or [])
    reference = spec.get("residual_reference_value")
    owned = {m for c in cfg["constituents"].values() for m in c["mechanisms"]}
    overlap = sorted(set(residuals) & owned)
    if overlap:
        raise ValueError(f"Profile mode '{name}' freezes mechanisms owned by a constituent: {overlap}")
    return {
        "profile_mode": name,
        "profile_mode_label": spec.get("label"),
        "residual_policy": spec.get("residual_policy"),
        "residual_mechanism_references": {m: float(reference) for m in residuals} if residuals and reference is not None else {},
        "residual_reference_label": spec.get("residual_reference_label"),
    }


def available_profile_modes(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    cfg = config or load_experimental_profile_config()
    return {k: v.get("label", k) for k, v in dict((cfg.get("profile_modes") or {}).get("modes") or {}).items()}


def resolve_experimental_profile(
    construct_values: Optional[Dict[str, float]] = None,
    constituent_overrides: Optional[Dict[str, float]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    profile_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve one agent's seven-variable profile.

    Returns a JSON-serializable record: the gestalt inputs, the generated values, the explicit overrides, the resolved
    constituents (with the source of each), and the mechanism values those constituents own.
    """
    cfg = config or load_experimental_profile_config()
    constructs = {str(k): float(v) for k, v in dict(construct_values or {}).items() if v is not None}
    explicit = {}
    for key, value in dict(constituent_overrides or {}).items():
        if value is None or str(value).strip() == "":
            continue
        if key not in cfg["constituents"]:
            raise ValueError(f"Unknown experimental constituent '{key}'. Known: {sorted(cfg['constituents'])}")
        explicit[str(key)] = _clamp(float(value))
    generated: Dict[str, float] = {}
    resolved: Dict[str, float] = {}
    source: Dict[str, str] = {}
    for name, spec in cfg["constituents"].items():
        value = float(spec["default"])
        origin = "default"
        rule = spec.get("generated_by")
        if rule and rule["construct"] in constructs:
            construct = constructs[rule["construct"]]
            centered = (construct - 0.5) * 2.0 if rule.get("transform", "linear_centered") == "linear_centered" else construct
            value = _clamp(value + float(rule["weight"]) * centered)
            generated[name] = value
            origin = f"generated:{rule['construct']}"
        if name in explicit:
            value = explicit[name]
            origin = "explicit_override"
        resolved[name] = _clamp(value)
        source[name] = origin
    mechanisms: Dict[str, float] = {}
    for name, spec in cfg["constituents"].items():
        for mechanism in spec["mechanisms"]:
            mechanisms[mechanism] = resolved[name]
    mode_record = resolve_profile_mode(profile_mode, config=cfg)
    return {
        **mode_record,
        "schema": cfg.get("schema"),
        "status": cfg.get("status"),
        "gestalt_constructs": constructs,
        "generated_constituents": generated,
        "explicit_constituent_overrides": explicit,
        "resolved_constituents": resolved,
        "constituent_source": source,
        "constituent_mechanisms": mechanisms,
    }